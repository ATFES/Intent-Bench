"""MLU operator timing over a product-owned queue and prepared calls."""
import ctypes as C
import torch
from ..model import PreparedCall
from intent.integrations.torch import torch_dtype
from intent.language.dtypes import dtype


def prepare_sequence(sequence, outputs):
    def decode(value):
        if isinstance(value, tuple):
            return tuple(decode(item) for item in value)
        if isinstance(value, list):
            return [decode(item) for item in value]
        if isinstance(value, dict):
            return {key: decode(item) for key, item in value.items()}
        if not callable(getattr(value, "to_host", None)):
            return value
        if not value.nbytes:
            return torch.empty(value.shape, dtype=torch_dtype(dtype(value.dtype)))
        return torch.frombuffer(value.to_host(), dtype=torch_dtype(dtype(value.dtype))).reshape(value.shape).clone()

    def measure(settings):
        queue = sequence.queue
        runtime = queue.runtime
        notifiers = []
        try:
            for _ in range(2):
                notifier = C.c_void_p()
                runtime.invoke("cnrtNotifierCreate", C.byref(notifier))
                notifiers.append(notifier)
            for _ in range(settings.warmup):
                sequence.reset()
                sequence.launch()
            times = []
            for _ in range(settings.samples):
                sequence.reset()
                runtime.invoke("cnrtPlaceNotifier", notifiers[0], queue.handle)
                sequence.enqueue()
                runtime.invoke("cnrtPlaceNotifier", notifiers[1], queue.handle)
                queue.synchronize()
                elapsed = C.c_float()
                runtime.invoke("cnrtNotifierDuration", *notifiers, C.byref(elapsed))
                times.append(elapsed.value*1e-3)
            return times
        finally:
            for notifier in notifiers:
                runtime.invoke("cnrtNotifierDestroy", notifier)
    return PreparedCall(sequence.launch, lambda:decode(outputs), reset=sequence.reset,
                        compile=sequence.compile, close=sequence.close, native_measure=measure,
                        device=f"mlu:{sequence.device}", timing="cnrt_notifier", owners=sequence,
                        metadata={"execution":"local_jit", "scope_note":"Complete authored sequence; transfers and state restoration excluded"})
