"""Compare installed author programs using their own input construction.

The registered programs each submit one complete call. Kernel definitions,
constexprs and multi-kernel composition remain in intent.examples; this module
only binds comparison implementations and observes their declared outputs.
"""
from ..model import Case, EnvironmentUnavailable, Implementation, PreparedCall, Workload
from dataclasses import dataclass


BASELINES = {
    "relu": {"cutile": "NVIDIA TileGym ReLU", "torch": "ATen ReLU", "cnnl": "CNNL ReLU"},
    "softmax": {"triton": "Triton fused-softmax tutorial", "torch": "ATen softmax", "cnnl": "CNNL softmax"},
    "transpose": {"torch": "ATen transpose materialization"},
    "embedding": {"triton": "Liger embedding lookup", "torch": "ATen embedding lookup"},
    "layer_norm": {"triton": "Triton layer-norm tutorial", "torch": "ATen LayerNorm",
                   "mojo": "MAX rowwise LayerNorm moments"},
    "gemm": {"triton": "Triton matrix-multiplication tutorial", "torch": "ATen GEMM", "cnnl": "CNNL matmul"},
    "batched_gemm": {"cutile": "NVIDIA TileGym batched GEMM", "torch": "ATen batched GEMM"},
    "histogram": {"cub": "NVIDIA CUB DeviceHistogram, including initialization and merge"},
    "swiglu": {"triton": "Liger fused SwiGLU", "torch": "ATen SiLU and multiply composition"},
    "rope": {"triton": "Liger RoPE, including layout conversion and InOut writeback"},
    "jagged_mean": {"triton": "TritonBench jagged mean", "torch": "ATen segment_reduce"},
    "cumsum": {"torch": "ATen cumsum"},
    "batch_norm": {"torch": "ATen native_batch_norm, including running-state updates"},
    "group_norm_backward": {"torch": "ATen native_group_norm_backward"},
    "adamw": {"torch": "ATen fused AdamW"},
    "cholesky": {"torch": "ATen/cuSOLVER Cholesky, InOut"},
    "dropout": {"cutile": "NVIDIA TileGym xorshift dropout"},
    "attention": {"cutile": "NVIDIA TileGym causal grouped-query FMHA"},
    "paged_decode": {"triton": "vLLM split-K paged GQA decode, including merge"},
    "mamba_chunk_state": {"triton": "state-spaces/mamba chunk state, FP32 states"},
    "gated_delta": {"cutile": "NVIDIA TileGym recurrent gated delta, including final state"},
}
from ..implementations.aten_cpu import OPERATIONS as CPU_PROGRAMS
for _name in BASELINES.keys() & CPU_PROGRAMS.keys():
    BASELINES[_name]["aten:cpu"] = "Native libtorch/ATen CPU " + _name


def backends(name):
    result = ["triton", "cutile"]
    if "mojo" in BASELINES[name] or "aten:cpu" in BASELINES[name]: result.append("mojo")
    if "cnnl" in BASELINES[name]: result.append("bangc")
    return result


def _tree(value, leaf):
    if isinstance(value, tuple):
        return tuple(_tree(item, leaf) for item in value)
    if isinstance(value, list):
        return [_tree(item, leaf) for item in value]
    if isinstance(value, dict):
        return {key: _tree(item, leaf) for key, item in value.items()}
    return leaf(value)


def _capture_results(value, getter, bindings):
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _capture_results(item, lambda index=index: getter()[index], bindings)
    elif isinstance(value, dict):
        for key, item in value.items():
            _capture_results(item, lambda key=key: getter()[key], bindings)
    else:
        bindings[id(value)] = getter


@dataclass
class _Definition:
    definition: object


def _mutable_arguments(program):
    from intent.language.annotations import ConstexprSpec, ViewKind, ViewSpec
    if not isinstance(program, _Definition):
        # Product compositions used here have read-only public inputs; their
        # workspace initialization belongs to the full prepared invocation.
        from intent.examples.programs.composition import Histogram, GroupNormBackward, PagedDecode
        if isinstance(program, (Histogram, GroupNormBackward, PagedDecode)):
            return ()
        raise NotImplementedError("declare public effects for this author composition")
    parameters = [p.annotation for p in program.definition.signature.parameters.values()
                  if not isinstance(p.annotation, ConstexprSpec)
                  and not (isinstance(p.annotation, ViewSpec) and p.annotation.kind is ViewKind.OUT)]
    return tuple(i for i, p in enumerate(parameters)
                 if isinstance(p, ViewSpec) and p.kind is ViewKind.INOUT)


def _device(env, backend=None):
    device = ("cpu" if backend in {"mojo", "aten:cpu"} else
              "mlu:0" if backend in {"bangc", "cnnl"} else "cuda:0") if env.device == "auto" else env.device
    if backend in {"triton", "cutile", "cub"} and not device.startswith("cuda:"):
        raise EnvironmentUnavailable(f"{backend} requires a CUDA execution environment")
    if backend in {"mojo", "aten:cpu"} and device != "cpu":
        raise EnvironmentUnavailable(f"{backend} requires a CPU execution environment")
    if backend in {"bangc", "cnnl"} and not device.startswith("mlu:"):
        raise EnvironmentUnavailable(f"{backend} requires an MLU execution environment")
    return device


def _seed(name, seed):
    from intent.examples import PROGRAMS
    from intent.examples.programs import inputs
    inputs.seed([seed, list(PROGRAMS).index(name)])


def _intent(name, backend, seed, env):
    from ..loading import intent_environment
    intent_environment(env)
    from intent.examples import load_program
    from intent.examples.programs.common import ExampleContext
    from ..implementations.intent import backend_options

    class PrepareContext(ExampleContext):
        invocation = None

        def call(self, program, *arguments):
            if self.invocation is not None:
                raise NotImplementedError("register complete multi-call author orchestration before benchmarking it")
            self.buffers.begin()
            self.invocation = program.prepare(*arguments)
            return self.invocation.result()

    context = PrepareContext(backend, _device(env, backend), compiler=env.compiler, stage="native",
                             workers=env.threads if backend == "mojo" else None,
                             backend_options=backend_options(env, backend))
    try:
        _seed(name, seed)
        output = load_program(name)(context)
        if context.invocation is None:
            raise ValueError("author program did not submit its complete call")
        call = context.invocation
        from intent.examples.programs.inputs import Array
        # Author returns can include original InOut arrays alongside Out values.
        # Resolve those arrays to the same resident buffers already bound above.
        output = _tree(output, lambda value: context.buffers.input(value, element=value.dtype)
                       if isinstance(value, Array) else value)
        restore = tuple(context.buffers.resetters.values())
        metadata = {"author_program": name, "input_profile": "installed author program",
                    "compiler_artifacts": [str(a.cache_directory) for a in context.artifacts]}
        reset = lambda: [function() for function in restore]
        if backend == "bangc":
            from types import SimpleNamespace
            from intent.runtime.bangc import Queue
            from ..implementations.bangc import prepare_sequence
            queue = Queue(device=int(context.device.split(':')[1]),
                          neuware=env.options.get("neuware", "/usr/local/neuware"))
            def close():
                try:
                    queue.close()
                finally:
                    context.close()
            sequence = SimpleNamespace(queue=queue, device=queue.device, reset=reset,
                enqueue=lambda: call.enqueue(queue), launch=call.launch, compile=call.compile, close=close)
            prepared = prepare_sequence(sequence, output)
            prepared.metadata.update(metadata)
            return prepared
        native_measure = None
        if backend == "mojo":
            from ..implementations.prepared import intent_sequence
            components = {"histogram": ("partials", "merge"), "group_norm_backward": ("dx", "affine")}
            calls = tuple(getattr(call, part) for part in components[name]) if name in components else (call,)
            native = intent_sequence(calls, lambda: output, reset=reset, metadata=metadata)
            def close():
                try:
                    native.close()
                finally:
                    context.close()
            from dataclasses import replace
            return replace(native, close=close, owners=context,
                compiler_observations=lambda: [a.observation for k in context.kernels for a in k.artifacts
                                                if a.observation is not None])
        return PreparedCall(call.launch, lambda: output, compile=call.compile,
            reset=reset, close=context.close, native_measure=native_measure,
            timing="cpu_native_monotonic" if native_measure else None,
            device=context.device, owners=context,
            metadata=metadata,
            compiler_observations=lambda: [a.observation for k in context.kernels for a in k.artifacts
                                            if a.observation is not None])
    except Exception:
        context.close()
        raise


def _reference(name, seed, env, implementation="torch"):
    import torch
    from intent.examples import load_program
    from intent.examples.programs.inputs import Array
    from ..corpus._common.measurement import functional_launch

    device = _device(env, implementation)
    if implementation in {"cnnl", "oracle"} and device.startswith("mlu:"):
        device = "cpu"
    dtypes = {"f16": torch.float16, "f32": torch.float32, "bf16": torch.bfloat16,
              "i32": torch.int32, "i64": torch.int64}
    buffers, results = {}, {}

    def tensor(value):
        if not isinstance(value, Array):
            return value
        if id(value) in buffers:
            return buffers[id(value)]
        # Preserve BF16 storage bits; NumPy's BF16 is not a torch.from_numpy dtype.
        data = torch.frombuffer(value.data, dtype=dtypes[value.dtype]).reshape(value.shape)
        buffers[id(value)] = data.to(device)
        return buffers[id(value)]

    class ReferenceContext:
        invocation = None

        def compile(self, definition, *, constexprs=None):
            # Author construction still describes its composition; the independent
            # library call does not compile or interpret Intent kernel definitions.
            return _Definition(definition)

        def bind(self, call, arguments, mutable):
            self.invocation = call
            initial = []
            for index in mutable:
                value = arguments[index]
                if value.device.type == "cpu":
                    import numpy as np
                    contents = value.detach().view(torch.uint8).numpy()
                    saved = contents.copy()
                    initial.append(lambda contents=contents, saved=saved: np.copyto(contents, saved))
                else:
                    saved = value.clone()
                    initial.append(lambda value=value, saved=saved: value.copy_(saved))
            if initial:
                call.reset = lambda: [restore() for restore in initial]
            # Some library APIs allocate their output lazily. One unmeasured
            # invocation establishes the author's return-tree projections.
            try:
                call.launch()
            except Exception:
                if call.close:
                    call.close()
                raise
            output = call.result()
            _capture_results(output, call.result, results)
            return output

        def call(self, program, *arguments):
            if self.invocation is not None:
                raise NotImplementedError("register the full author sequence before benchmarking it")
            arguments = tuple(tensor(value) for value in arguments)
            mutable = _mutable_arguments(program)
            if implementation == "aten:cpu":
                from ..implementations.aten_cpu import prepare
                return self.bind(prepare(name, arguments, env), arguments, mutable)
            if implementation == "cnnl":
                from ..implementations.native_libraries import cnnl
                operation = "matmul" if name == "gemm" else name
                call = cnnl(operation, None, tuple(value.numpy() for value in arguments), env)
                return self.bind(call, arguments, mutable)
            if name == "histogram":
                from ..implementations.cub import histogram
                return self.bind(histogram(arguments[0], env), arguments, mutable)
            if implementation == "mojo":
                from types import SimpleNamespace
                from ..corpus.cpu.baselines.mojo.modular.normalization.layer_norm.layer_norm_runtime import prepare
                from ..implementations.native_sampling import sample_timer
                native = prepare(SimpleNamespace(workers=env.threads, executable=env.options.get("mojo")),
                                 *arguments)
                metadata = {"library": "MAX rowwise LayerNorm moments",
                            "reference_source": "corpus/cpu/baselines/mojo/modular/normalization/layer_norm/layer_norm.mojo"}
                call = PreparedCall(native.launch, native.result, device=device,
                    timing="cpu_native_monotonic", owners=native, metadata=metadata,
                    native_measure=lambda settings: sample_timer(native, settings, lambda: None, metadata=metadata))
                return self.bind(call, arguments, mutable)
            from ..references.author_programs import prepare as prepare_additional
            if name in {"attention", "paged_decode", "mamba_chunk_state", "gated_delta"}:
                from ..references.author_streaming import prepare
                return self.bind(prepare(name, implementation, arguments, env), arguments, mutable)
            if name in {"swiglu", "rope", "jagged_mean", "cumsum", "batch_norm",
                        "group_norm_backward", "adamw", "cholesky", "dropout"}:
                return self.bind(prepare_additional(name, implementation, arguments, env), arguments, mutable)
            x = arguments[0]
            source = "PyTorch ATen "+("CUDA" if device.startswith("cuda:") else "CPU")
            if implementation == "triton":
                from ..corpus import PROJECT_ROOT
                from ..corpus._common.loading import load_module
                paths = {
                    "softmax": "triton/normalization/softmax/02-fused-softmax_runtime.py",
                    "layer_norm": "triton/normalization/layer_norm/05-layer-norm_runtime.py",
                    "gemm": "triton/gemm/dense/03-matrix-multiplication_runtime.py",
                    "embedding": "liger-kernel/embedding/lookup/embedding_runtime.py",
                }
                path = PROJECT_ROOT/"corpus/gpu/baselines/triton"/paths[name]
                runtime = load_module(path, "author_reference_"+name)
                if name == "softmax":
                    native = runtime.load_softmax()
                    function = lambda: native(x)
                elif name == "gemm":
                    native = runtime.load_matmul()
                    function = lambda: native(*arguments)
                elif name == "layer_norm":
                    native = runtime.load_layer_norm()
                    function = lambda: native(x, (x.shape[-1],), arguments[1], arguments[2], arguments[4])
                else:
                    function = lambda: runtime.LigerEmbeddingFunction.apply(*arguments)
                source = str(path)
            elif implementation == "cutile":
                from ..corpus import PROJECT_ROOT
                from ..implementations.context import Context
                from ..source_support import tilegym_source
                context = Context(env.compiler, PROJECT_ROOT, None, "cutile", environment=env)
                path = ("activation/relu/relu.py" if name == "relu" else "gemm/batched/bmm.py")
                module = tilegym_source(context, "corpus/gpu/baselines/cutile/tilegym/"+path, "author_"+name)
                function = (lambda: module.relu(x)) if name == "relu" else (
                    lambda: module.bmm(*arguments, transpose_a=False, transpose_b=False,
                                       static_persistent=True, compiler_timeout=60))
                source = "NVIDIA TileGym/"+path
            elif name == "relu":
                function = lambda: torch.relu(x)
            elif name == "softmax":
                function = lambda: torch.softmax(x, dim=-1)
            elif name == "transpose":
                function = lambda: x.t().contiguous()
            elif name == "embedding":
                function = lambda: torch.nn.functional.embedding(arguments[1], x)
            elif name == "layer_norm":
                function = lambda: torch.nn.functional.layer_norm(
                    x, (x.shape[-1],), arguments[1], arguments[2], arguments[4])
            elif name == "gemm":
                function = lambda: torch.mm(x, arguments[1])
            elif name == "batched_gemm":
                function = lambda: torch.bmm(x, arguments[1])
            else:
                raise NotImplementedError(name)
            call = functional_launch(function)
            call.device = device
            call.owners = arguments
            call.metadata["library"] = source
            return self.bind(call, arguments, mutable)

    _seed(name, seed)
    context = ReferenceContext()
    output = load_program(name)(context)
    if context.invocation is None:
        raise ValueError("author program did not submit its complete call")
    context.invocation.metadata.update(author_program=name, input_profile="installed author program")
    from dataclasses import replace
    def observe(value):
        if isinstance(value, Array):
            return buffers[id(value)]
        return results[id(value)]() if id(value) in results else value
    return replace(context.invocation, result=lambda: _tree(output, observe))


def build(name):
    if name not in BASELINES:
        raise KeyError(name)
    from ..implementations.context import Tolerance
    from ..numeric import compare_outputs
    tolerances = {"softmax": Tolerance(.01, .01), "layer_norm": Tolerance(5e-5, 1e-5),
                  "gemm": Tolerance(.01, .01), "batched_gemm": Tolerance(.05, .02),
                  "swiglu": Tolerance(.02, .01), "rope": (Tolerance(.002, .002),)*2,
                  "jagged_mean": Tolerance(2e-5, 1e-5), "cumsum": Tolerance(.002, 1e-5),
                  "group_norm_backward": Tolerance(.05, .02), "adamw": Tolerance(1e-5, 1e-5),
                  "cholesky": Tolerance(.0002),
                  "attention": Tolerance(.05, .02), "paged_decode": Tolerance(.03),
                  "mamba_chunk_state": Tolerance(.02, .02),
                  "gated_delta": (Tolerance(.1, .05),)*2,
                  "batch_norm": (Tolerance(.02, .01), *((Tolerance(.0001, .0001),)*4))}
    tolerance = tolerances.get(name, Tolerance(0))
    implementations = {baseline: Implementation(baseline,
        lambda case, seed, env, baseline=baseline: _reference(name, seed, env, baseline), description)
        for baseline, description in BASELINES[name].items()}
    for backend in backends(name):
        implementations["intent:"+backend] = Implementation("intent:"+backend,
            lambda case, seed, env, backend=backend: _intent(name, backend, seed, env))
    # Histogram's independent CPU oracle also checks excluded bins in the author
    # input; its output is integer, so no tolerance is applied to counts.
    def histogram_oracle(case, seed):
        import numpy as np
        import torch
        from intent.examples import load_program
        class Context:
            def compile(self, *args, **kwargs): return None
            def call(self, program, samples):
                indices = samples.data.astype(np.int32)
                counts = np.bincount(indices[(indices >= 0) & (indices < 256)], minlength=256).astype(np.int32)
                return torch.from_numpy(counts)
        _seed(name, seed)
        return load_program(name)(Context())
    oracle = histogram_oracle if name == "histogram" else Implementation(
        "oracle", lambda case, seed, env: _reference(name, seed, env, "oracle"))
    def check(actual, expected):
        import numpy as np
        import torch
        tensor = lambda value: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
        actual, expected = _tree(actual, tensor), _tree(expected, tensor)
        if name == "histogram":
            expected = expected.to(actual.device)
        compare_outputs(actual, expected, tolerance)
    return Workload("program/"+name, [Case("author")], lambda case, seed: seed,
        implementations, oracle=oracle, check=check,
        family="program", parameter_keys=())
