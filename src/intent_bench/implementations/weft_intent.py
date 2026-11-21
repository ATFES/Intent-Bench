"""Bind the installed Weft product on the executing CPU."""
from contextlib import ExitStack
import numpy as np
from ..loading import load
from .intent import make_kernel
from .prepared import intent_sequence


def prepare(reference, args, env):
    from intent import HostBuffers
    from intent.examples.programs.inputs import numpy_dtype

    def decode(value):
        if isinstance(value, tuple):
            return tuple(decode(item) for item in value)
        if value is None:
            return None
        return np.frombuffer(value.to_host(), dtype=numpy_dtype(value.dtype)).reshape(value.shape)

    owners = ExitStack()
    try:
        kernel = make_kernel(load(reference), env, "weft")
        owners.callback(kernel.close)
        buffers = owners.enter_context(HostBuffers("weft", kernel.device, kernel.backend_options))
        arguments = tuple(buffers.input(value, preserve=True) if isinstance(value, np.ndarray)
                          else value for value in args)
        call = kernel.prepare(*arguments)
        metadata = {"execution": "local_jit",
                    "compiler_artifacts": [str(artifact.cache_directory) for artifact in kernel.artifacts]}
        prepared = intent_sequence(call, lambda: decode(call.result()), reset=buffers.reset,
                               metadata=metadata, release=owners.close)
        from .prepared import native_selection
        prepared.tuning_report=lambda: native_selection(call,kernel)
        return prepared
    except BaseException:
        owners.close()
        raise
