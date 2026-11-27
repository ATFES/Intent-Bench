"""Independent kernels for the installed programs' mutable/compound contracts."""
from ..model import PreparedCall


def prepare(name, implementation, arguments, environment):
    import torch
    from ..corpus import PROJECT_ROOT
    from ..corpus._common.loading import load_module
    from ..corpus._common.measurement import functional_launch
    from ..corpus.cpu.baselines.pytorch import cpu_runtime

    x = arguments[0]
    library = "PyTorch ATen CUDA"
    note = ""
    if implementation == "triton":
        paths = {
            "swiglu": "liger-kernel/activation/swiglu/swiglu_runtime.py",
            "rope": "liger-kernel/position/rope/rope_runtime.py",
            "jagged_mean": "tritonbench/ragged/jagged_mean/jagged_mean_runtime.py",
        }
        path = PROJECT_ROOT/"corpus/gpu/baselines/triton"/paths[name]
        module = load_module(path, "author_reference_"+name)
        if name == "rope":
            native = module.load_rope_forward()
            def run():
                q, k, _, _ = native(*arguments)
                arguments[0].copy_(q)
                arguments[1].copy_(k)
                return arguments[:2]
            note = "Liger requires sequence-major storage; its layout conversions and InOut writeback are included."
        else:
            run = lambda: module.upstream(arguments)
        library = str(path)
    elif implementation == "cutile":
        from types import SimpleNamespace
        from ..implementations.context import Context
        from ..source_support import tilegym_source
        context = Context(environment.compiler, PROJECT_ROOT, None, "cutile", environment=environment)
        module = tilegym_source(context,
            "corpus/gpu/baselines/cutile/tilegym/regularization/dropout/dropout.py", "author_dropout")
        values, seed, probability, inverse = arguments
        if inverse != 1/(1-probability):
            raise ValueError("native dropout requires inverse_keep_probability = 1/(1-p)")
        output = torch.empty_like(values)
        configs = tuple(SimpleNamespace(TILE_SIZE=1 << exponent, ACCESS_FORM=form)
                        for exponent in range(5, 15) for form in (1, 2, 3))
        def run():
            # The author already supplies the mixed seed; call the existing
            # primitive directly instead of applying the wrapper's mix twice.
            module._autotune_dropout(torch.cuda.current_stream(values.device), values.reshape(-1),
                output.reshape(-1), probability, seed, configs, 60)
            return output
        library = "NVIDIA TileGym dropout kernel"
        note = "Author mixed seed passed directly; same xorshift mask and f16 scale."
    elif name == "swiglu":
        run = lambda: (torch.nn.functional.silu(x.float()) * arguments[1].float()).to(x.dtype)
    elif name == "jagged_mean" and implementation == "aten:source":
        offsets = arguments[1]
        def run():
            lengths = offsets.diff()
            rows = torch.arange(lengths.numel(), device=x.device)
            indices = rows.repeat_interleave(lengths.long(), output_size=x.shape[0])
            return x.new_zeros((lengths.numel(), x.shape[1])).index_add_(0, indices, x) / lengths[:, None]
        library = "ATen segmented sum and quotient"
    elif name == "jagged_mean":
        run = lambda: torch.segment_reduce(x, "mean", offsets=arguments[1])
    elif name == "cumsum":
        run = lambda: torch.cumsum(x, dim=-1)
    elif name == "batch_norm":
        if implementation in {"aten:source", "oracle"}:
            library = "ATen variance and mean, training state updates"
            run = lambda: cpu_runtime.batch_norm_training(*arguments)[2:]
        else:
            run = lambda: torch.ops.aten.native_batch_norm(*arguments[:5], True,
                                                           arguments[6], arguments[5])
            library = "ATen native_batch_norm (training, saved statistics, running-state updates)"
    elif name == "group_norm_backward":
        if implementation == "oracle":
            run = lambda: cpu_runtime.group_norm_backward(*arguments[:5])
        else:
            values, gradient, weight, mean, rstd, _ = arguments
            run = lambda: torch.ops.aten.native_group_norm_backward(
                gradient, values, mean, rstd, weight, *values.shape,
                mean.shape[1], [True, True, True])
            library = "ATen native_group_norm_backward"
    elif name == "adamw":
        if implementation in {"oracle", "aten:source"}:
            library = "ATen FP32 AdamW, three source-order quotients"
            # Constants in this program are FP32. Keep them on device so
            # (1 - beta) rounds in FP32 and TensorIterator does not turn a
            # wrapped CPU scalar divisor into a host-computed reciprocal.
            values = arguments[:4] + tuple(x.new_tensor(v) for v in arguments[4:])
            run = lambda: cpu_runtime.adamw(*values)
        else:
            gradient, parameter, first, second, lr, beta1, beta2, correction1, correction2, eps, decay = arguments
            if correction1 != 1-beta1 or correction2 != 1-beta2:
                raise ValueError("this author AdamW case requires a first-step bias correction")
            step = torch.ones((), device=parameter.device, dtype=torch.float32)
            def run():
                torch.ops.aten._fused_adamw_([parameter], [gradient], [first], [second], [], [step],
                    lr=lr, beta1=beta1, beta2=beta2, weight_decay=decay, eps=eps,
                    amsgrad=False, maximize=False)
                return parameter, first, second
            library = "ATen _fused_adamw_"
    elif name == "cholesky":
        info = torch.empty(x.shape[:-2], dtype=torch.int32, device=x.device)
        def run():
            torch.linalg.cholesky_ex(x, check_errors=False, out=(x, info))
            return x
        library = "ATen/cuSOLVER linalg_cholesky_ex (InOut)"
        note = "Library info output is computed; host error checking is outside operator timing."
    elif name == "rope":
        def run():
            for value in arguments[:2]:
                half = value.shape[-1]//2
                left, right = value[..., :half].clone(), value[..., half:].clone()
                cosine, sine = arguments[2][..., :half], arguments[3][..., :half]
                value[..., :half].copy_(left*cosine-right*sine)
                value[..., half:].copy_(right*cosine+left*sine)
            return arguments[:2]
        library = "Independent RoPE numerical oracle"
    elif name == "dropout":
        run = lambda: cpu_runtime.dropout(x, *arguments[1:])
        library = "ATen signed xorshift and FP16 scale"
    else:
        raise NotImplementedError((name, implementation))
    call = functional_launch(run)
    call.device = str(x.device)
    if implementation == "aten:source" and call.device == "cpu":
        from ..implementations.prepared import attach_device_timer
        call = attach_device_timer(call)
    call.owners = arguments
    call.metadata.update(library=library, note=note)
    return call
