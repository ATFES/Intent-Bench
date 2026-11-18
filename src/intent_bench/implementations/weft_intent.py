"""Use the installed product on the executing RISC-V host."""
import numpy as np
from ..loading import load
from .intent import make_kernel
from .prepared import intent_sequence


def prepare(reference, args, env, *, ime=False):
    from intent.runtime.weft import Buffer
    from intent.examples.programs.inputs import numpy_dtype
    dtypes = {str(numpy_dtype(name)): name for name in
              ("bool", "u8", "i8", "u16", "i16", "u32", "i32", "u64", "i64", "f16", "f32", "f64")}
    arrays = [value for value in args if isinstance(value, np.ndarray)]
    saved = [value.copy() for value in arrays]
    arguments = tuple(Buffer(value, shape=value.shape, dtype=dtypes[str(value.dtype)])
                      if isinstance(value, np.ndarray) else value for value in args)
    kernel = make_kernel(load(reference), env, "weft")
    call = kernel.prepare(*arguments)
    def restore():
        for value, initial in zip(arrays, saved, strict=True):
            np.copyto(value, initial)
    def decode(value):
        if isinstance(value, tuple):
            return tuple(decode(item) for item in value)
        if value is None:
            return None
        return np.frombuffer(value.storage, dtype=numpy_dtype(value.dtype)).reshape(value.shape).copy()
    metadata = {"execution": "local_jit",
                "compiler_artifacts": [str(artifact.cache_directory) for artifact in kernel.artifacts]}
    return intent_sequence(call, lambda: decode(call.result()), reset=restore,
                           metadata=metadata, release=kernel.close)
