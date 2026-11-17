"""MLU operator timing over a product-owned queue and prepared calls."""
import ctypes as C
from ..model import PreparedCall
from .outputs import native_result


def prepare_sequence(sequence, outputs, *, environment, reset=None, close=None, owners=None):
    from intent.runtime.bangc import Queue
    queue = Queue(device=int(environment.device.split(':')[1]),
                  neuware=environment.options.get('neuware', '/usr/local/neuware'))
    restore = reset or (lambda: None)
    def release():
        try:
            queue.close()
        finally:
            if close is not None:
                close()
    def measure(settings):
        runtime = queue.runtime
        notifiers = []
        try:
            for _ in range(2):
                notifier = C.c_void_p()
                runtime.invoke("cnrtNotifierCreate", C.byref(notifier))
                notifiers.append(notifier)
            for _ in range(settings.warmup):
                restore()
                sequence.launch()
            times = []
            for _ in range(settings.samples):
                restore()
                runtime.invoke("cnrtPlaceNotifier", notifiers[0], queue.handle)
                sequence.enqueue(queue)
                runtime.invoke("cnrtPlaceNotifier", notifiers[1], queue.handle)
                queue.synchronize()
                elapsed = C.c_float()
                runtime.invoke("cnrtNotifierDuration", *notifiers, C.byref(elapsed))
                times.append(elapsed.value*1e-3)
            return times
        finally:
            for notifier in notifiers:
                runtime.invoke("cnrtNotifierDestroy", notifier)
    return PreparedCall(sequence.launch, lambda:native_result(outputs), reset=restore,
                        compile=sequence.compile, close=release, native_measure=measure,
                        device=f"mlu:{queue.device}", timing="cnrt_notifier", owners=(sequence, owners),
                        metadata={"execution":"local_jit", "scope_note":"Complete authored sequence; transfers and state restoration excluded"})


def prepare_program(context, program, arguments=None, *, mutable_positions=(), result=None):
    """Stage declared inputs; binding and the complete sequence belong to Intent."""
    import intent
    import numpy as np
    import torch
    environment = context.environment
    buffers = intent.HostBuffers(context.provider, environment.device,
        environment.options.get('intent', {}).get(context.provider, {}))
    arguments = context.inputs if arguments is None else arguments
    native = tuple(buffers.input(value, writable=index in mutable_positions,
                                 preserve=index in mutable_positions)
                   if isinstance(value, (torch.Tensor, np.ndarray)) else value
                   for index, value in enumerate(arguments))
    try:
        call = program.prepare(*native)
        buffers.retain_outputs(call.result())
        output = call.result() if result is None else result(call, native)
    except Exception:
        buffers.close()
        raise
    def close():
        try:
            for kernel in context.kernels:
                kernel.close()
        finally:
            buffers.close()
    return prepare_sequence(call, output, environment=environment, reset=buffers.reset,
                            close=close, owners=(program, buffers))
