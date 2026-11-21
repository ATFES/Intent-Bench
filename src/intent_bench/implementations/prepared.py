"""Bind an Intent artifact once; construct Python integration only on explicit use."""
from ..model import PreparedCall


def checked_call(call, tolerance, *, cuda_graph=False, device_type='cuda', note='', status='pass', owners=None):
    """Attach a workload's numerical contract to its one prepared invocation."""
    if call is None:
        from ..model import ExpressionGap
        raise ExpressionGap(note or status)
    if not isinstance(call, PreparedCall):
        raise TypeError("implementation preparation must return PreparedCall")
    call.tolerance = tolerance
    call.owners = owners
    if call.device == 'cpu' and device_type == 'cuda':
        call.device = 'cuda'
    call.metadata['note'] = note
    return call


def native_samples(measure):
    def sample(settings):
        for _ in range(settings.warmup):
            measure()
        return [measure() for _ in range(settings.samples)]
    return sample


def intent_call(artifact, arguments, *, device='cpu', reset=None, observe=None,
                host_factory=None, bindings=None):
    bindings = {} if bindings is None else bindings
    call = artifact.prepare(*arguments, **bindings)
    observe = observe or (lambda result: result)
    host = {}

    def host_run():
        if 'invoke' not in host:
            host['invoke'] = host_factory() if host_factory else lambda: artifact.run(*arguments, **bindings)
        host['output'] = host['invoke']()

    metadata = {'artifacts': [str(a.cache_directory) for a in getattr(artifact, 'artifacts', ())],
                'operator_entry': 'prepared_launch',
                'host_entry': 'torch_compile_custom_op' if host_factory else 'artifact.run'}
    if device == 'cpu':
        return intent_sequence(call, lambda: observe(call.result()), reset=reset,
                               host_run=host_run, host_result=lambda: observe(host['output']),
                               metadata=metadata)
    def compile_native():
        call.compile()
        snap = getattr(call, "observation", None)
        if snap is not None and snap.provider == "cutile":
            from dataclasses import asdict
            rows = [asdict(c) for c in snap.candidates]
            metadata["cold_compile"] = {"candidate_count": len(rows), "rows": rows,
                "instrumentation": "cuTile bound native compilation observations"}
    return PreparedCall(
        launch=call.launch, result=lambda: observe(call.result()), compile=compile_native,
        reset=reset, device=device,
        metadata=metadata,
        host_run=host_run, host_result=lambda: observe(host['output']),
    )


class IntentSequence:
    """Bind outputs and workspace once for a complete multi-kernel CPU operator."""
    def __init__(self):
        self.calls = []
        self.artifacts = []

    def add(self, artifact, *arguments, **bindings):
        call = artifact.prepare(*arguments, **bindings)
        self.calls.append(call)
        self.artifacts.append(artifact)
        return call.result()

    def prepared(self, result, *, reset=None, host_run=None, host_result=None, initializations=()):
        from intent import PreparedSequence
        fills = tuple(self.artifacts[0].prepare_fill(buffer, value) for buffer, value in initializations)
        sequence = PreparedSequence((*fills, *self.calls), outputs=None, owners=self.artifacts)
        return intent_sequence(sequence, result, reset=reset, host_run=host_run, host_result=host_result,
                               metadata={'artifacts': [str(a.cache_directory) for kernel in self.artifacts for a in kernel.artifacts]})


def intent_sequence(call, result, *, reset=None, host_run=None, host_result=None, metadata=None, release=None):
    from intent import PreparedSequence
    if isinstance(call, (tuple, list)):
        call = PreparedSequence(call, outputs=None)
    native = {}
    restore = reset or (lambda: None)
    call_metadata = {**(metadata or {}), 'operator_entry': 'prepared_native_sequence',
                     'kernel_count': call.kernel_count if isinstance(call, PreparedSequence) else 1,
                     'initialization_count': call.initialization_count if isinstance(call, PreparedSequence) else 0}

    def measure(settings):
        if not native:
            from .native_sequence import NativeSequence
            native['timer'] = NativeSequence((call,))
        from .native_sampling import sample_timer
        call_metadata['cache_policy'] = 'flush_cpu_64m' if settings.cache == 'flush' else 'warm'
        return sample_timer(native['timer'], settings, restore)

    def close():
        if native:
            native['timer'].close()
        if release is not None:
            release()

    def observations():
        if isinstance(call, PreparedSequence):
            return call.observations
        current = call.observation
        return () if current is None else (current,)

    return PreparedCall(call.launch, result, compile=call.compile, reset=reset,
                        native_measure=measure, timing='cpu_native_monotonic', close=close,
                        host_run=host_run or call.launch, host_result=host_result or result,
                        metadata=call_metadata, owners=call, compiler_observations=observations)
