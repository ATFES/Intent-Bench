"""Compare installed author programs using their own input construction.

The registered programs each submit one complete call. Kernel definitions,
constexprs and multi-kernel composition remain in intent.examples; this module
only binds comparison implementations and observes their declared outputs.
"""
from ..model import Case, EnvironmentUnavailable, Implementation, PreparedCall, Workload


BASELINES = {
    "relu": ("cutile", "torch"), "softmax": ("triton", "torch"), "transpose": ("torch",),
    "embedding": ("triton", "torch"), "layer_norm": ("triton", "torch", "mojo"),
    "gemm": ("triton", "torch"), "batched_gemm": ("cutile", "torch"), "histogram": ("cub",),
}


def _device(env, backend=None):
    device = ("cpu" if backend == "mojo" else "cuda:0") if env.device == "auto" else env.device
    if backend in {"triton", "cutile", "cub"} and not device.startswith("cuda:"):
        raise EnvironmentUnavailable(f"{backend} requires a CUDA execution environment")
    if backend == "mojo" and device != "cpu":
        raise EnvironmentUnavailable("Mojo CPU comparison requires a CPU execution environment")
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

    context = PrepareContext(backend, _device(env, backend), compiler=env.compiler, stage="native",
                             workers=env.threads if backend == "mojo" else None,
                             backend_options=backend_options(env, backend))
    try:
        _seed(name, seed)
        load_program(name)(context)
        if context.invocation is None:
            raise ValueError("author program did not submit its complete call")
        call = context.invocation
        restore = tuple(context.buffers.resetters.values())
        metadata = {"author_program": name, "input_profile": "installed author program",
                    "compiler_artifacts": [str(a.cache_directory) for a in context.artifacts]}
        reset = lambda: [function() for function in restore]
        native_measure = None
        if backend == "mojo":
            from ..implementations.native_sampling import sample
            native_measure = lambda settings: sample(call, settings, reset, metadata=metadata)
        return PreparedCall(call.launch, call.result, compile=call.compile,
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
    dtypes = {"f16": torch.float16, "f32": torch.float32, "bf16": torch.bfloat16,
              "i32": torch.int32, "i64": torch.int64}

    def tensor(value):
        if not isinstance(value, Array):
            return value
        # Preserve BF16 storage bits; NumPy's BF16 is not a torch.from_numpy dtype.
        data = torch.frombuffer(value.data, dtype=dtypes[value.dtype]).reshape(value.shape)
        return data.to(device)

    class ReferenceContext:
        invocation = None

        def compile(self, definition, *, constexprs=None):
            # Author construction still describes its composition; the independent
            # library call does not compile or interpret Intent kernel definitions.
            return None

        def call(self, program, *arguments):
            if self.invocation is not None:
                raise NotImplementedError("register the full author sequence before benchmarking it")
            arguments = tuple(tensor(value) for value in arguments)
            if name == "histogram":
                from ..implementations.cub import histogram
                self.invocation = histogram(arguments[0], env)
                return
            if implementation == "mojo":
                from types import SimpleNamespace
                from ..corpus.cpu.baselines.mojo.modular.normalization.layer_norm.layer_norm_runtime import prepare
                from ..implementations.native_sampling import sample_timer
                native = prepare(SimpleNamespace(workers=env.threads, executable=env.options.get("mojo")),
                                 *arguments)
                metadata = {"library": "MAX rowwise LayerNorm moments",
                            "reference_source": "corpus/cpu/baselines/mojo/modular/normalization/layer_norm/layer_norm.mojo"}
                self.invocation = PreparedCall(native.launch, native.result, device=device,
                    timing="cpu_native_monotonic", owners=native, metadata=metadata,
                    native_measure=lambda settings: sample_timer(native, settings, lambda: None, metadata=metadata))
                return
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
            self.invocation = functional_launch(function)
            self.invocation.device = device
            self.invocation.owners = arguments
            self.invocation.metadata["library"] = source

    _seed(name, seed)
    context = ReferenceContext()
    load_program(name)(context)
    if context.invocation is None:
        raise ValueError("author program did not submit its complete call")
    context.invocation.metadata.update(author_program=name, input_profile="installed author program")
    return context.invocation


def build(name):
    if name not in BASELINES:
        raise KeyError(name)
    from ..implementations.context import Tolerance
    from ..numeric import compare_outputs
    tolerances = {"softmax": Tolerance(.01, .01), "layer_norm": Tolerance(5e-5, 1e-5),
                  "gemm": Tolerance(.01, .01), "batched_gemm": Tolerance(.05, .02)}
    tolerance = tolerances.get(name, Tolerance(0))
    implementations = {baseline: Implementation(baseline,
        lambda case, seed, env, baseline=baseline: _reference(name, seed, env, baseline))
        for baseline in BASELINES[name]}
    for backend in (("triton", "cutile", "mojo") if name == "layer_norm" else ("triton", "cutile")):
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
        "oracle", lambda case, seed, env: _reference(name, seed, env))
    def check(actual, expected):
        if name == "histogram":
            expected = expected.to(actual.device)
        compare_outputs(actual, expected, tolerance)
    return Workload("program/"+name, [Case("author")], lambda case, seed: seed,
        implementations, oracle=oracle, check=check,
        family="program", parameter_keys=())
