"""Bind the installed V2 kernel to the existing author's exact input buffers."""
from functools import partial

from ..model import Implementation, PreparedCall


CATALOG = {"rotary_embedding_equivalent_index"}


def registrations(name):
    if name not in CATALOG and name not in {"relu", "transpose", "swiglu", "dropout", "adamw", "rope"}:
        return {}
    return {
        "intent-v2:" + backend: Implementation(
            "intent-v2:" + backend, partial(prepare, name, backend)
        )
        for backend in ("triton", "cutile", "mojo", "weft", "bangc")
    }


def prepare(name, backend, case, values, env):
    from ..workloads.programs import _device

    device = _device(env, backend)
    if name in CATALOG:
        args = values.arguments
        if device != "cpu":
            if device.startswith("mlu:"):
                import torch_mlu
            args = tuple(v.to(device) for v in args)
        call = bind(name, backend, args, env, device)
        if device.startswith("cuda:"):
            result = call.result
            call.result = lambda: result().cpu()
        return call
    from intent.examples.programs.inputs import Array
    if device == "cpu":
        args = [x.to_numpy() if isinstance(x, Array) else x for x in values["arguments"]]
    else:
        import torch
        if device.startswith("mlu:"):
            import torch_mlu
        dtypes = {"f16": torch.float16, "bf16": torch.bfloat16,
                  "f32": torch.float32, "i32": torch.int32}
        args = [
            torch.frombuffer(x.data, dtype=dtypes[x.dtype]).reshape(x.shape).to(device)
            if isinstance(x, Array) else x for x in values["arguments"]
        ]
    return bind(name, backend, args, env, device)


def bind(name, backend, args, env, device):
    import intentdsl
    from intentdsl import examples
    options = env.options.get("compilerinfra", {})
    kernel = intentdsl.jit(getattr(examples, name), backend=backend,
                           library=options.get("library"), configuration=options.get("configuration"))
    mutable = {"adamw": (1,2,3), "rope":(0,1)}.get(name, ())
    original = [(args[i], args[i].copy() if device=="cpu" else args[i].clone()) for i in mutable]
    def reset():
        for dest, value in original:
            if device=="cpu":
                dest[...] = value
            else:
                dest.copy_(value)
    call = kernel.prepare(*args)
    if "row_id" in options:
        call = call.bind_row(options["row_id"])
    metadata = {"author_program": name}
    def compile():
        call.compile()
        if backend != "triton":
            metadata["cold_compile"] = call.compilation_report()

    prepared = PreparedCall(call.launch, call.result, device=device, compile=compile,
                            tuning_report=call.tuning_report, reset=reset if original else None, metadata=metadata)
    if device == "cpu":
        from .native_sequence import NativeSequence
        from .native_sampling import sample_timer

        timer = None
        def measure(settings):
            nonlocal timer
            if timer is None:
                timer = NativeSequence((call,))
            return sample_timer(timer, settings, reset, metadata=metadata)
        prepared.native_measure = measure
        prepared.close = lambda: timer and timer.close()
        prepared.timing = "cpu_native_monotonic"
    elif device.startswith("mlu:"):
        from .mlu_rows import timer
        import torch
        stream = torch.mlu.current_stream(device)
        prepared.synchronize = stream.synchronize
        prepared.native_measure = lambda settings: timer(call.launch, stream, settings, reset)
        prepared.timing = "cnrt_notifier"
        from ..workloads.programs import _tree
        prepared.result = lambda: _tree(call.result(), lambda v:v.cpu())
    return prepared
