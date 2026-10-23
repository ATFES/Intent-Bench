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


def intent_call(artifact, arguments, *, device='cpu', reset=None, observe=None, host_factory=None):
    call = artifact.prepare(*arguments)
    observe = observe or (lambda result: result)
    host = {}

    def host_run():
        if 'invoke' not in host:
            host['invoke'] = host_factory() if host_factory else lambda: artifact.run(*arguments)
        host['output'] = host['invoke']()

    native_measure = None
    metadata = {'artifacts': [str(a.cache_directory) for a in artifact.artifacts],
                'operator_entry': 'prepared_launch',
                'host_entry': 'torch_compile_custom_op' if host_factory else 'artifact.run'}
    if device == 'cpu':
        from .native_sampling import sample
        def native_measure(settings):
            metadata['cache_policy'] = 'flush_cpu_64m' if settings.cache == 'flush' else 'warm'
            return sample(call, settings, reset or (lambda: None))
    return PreparedCall(
        launch=call.launch, result=lambda: observe(call.result()), compile=call.compile,
        reset=reset, device=device, native_measure=native_measure,
        timing='cpu_native_monotonic' if native_measure else None,
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
        return intent_sequence(self.calls, result, reset=reset, host_run=host_run, host_result=host_result,
                               metadata={'artifacts': [str(a.cache_directory) for kernel in self.artifacts for a in kernel.artifacts]},
                               initializations=initializations)


def intent_sequence(calls, result, *, reset=None, host_run=None, host_result=None, metadata=None,
                    initializations=()):
    calls = tuple(calls)
    initializations = tuple(initializations)
    native = {}
    restore = reset or (lambda: None)
    call_metadata = {**(metadata or {}), 'operator_entry': 'prepared_native_sequence',
                     'kernel_count': len(calls), 'initialization_count': len(initializations)}

    def run():
        for buffer, value in initializations:
            buffer.fill_(value)
        for call in calls:
            call.launch()

    def measure(settings):
        if not native:
            from .native_sequence import NativeSequence
            native['timer'] = NativeSequence(calls, initializations=initializations)
        from .native_sampling import sample_timer
        call_metadata['cache_policy'] = 'flush_cpu_64m' if settings.cache == 'flush' else 'warm'
        return sample_timer(native['timer'], settings, restore)

    def close():
        if native:
            native['timer'].close()

    return PreparedCall(run, result, compile=lambda: [call.compile() for call in calls], reset=reset,
                        native_measure=measure, timing='cpu_native_monotonic', close=close,
                        host_run=host_run or run, host_result=host_result or result,
                        metadata=call_metadata)
