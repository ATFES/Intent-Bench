"""Observe public native buffers after execution, preserving storage dtype."""


def native_result(value):
    if isinstance(value, tuple):
        return tuple(native_result(item) for item in value)
    if isinstance(value, list):
        return [native_result(item) for item in value]
    if isinstance(value, dict):
        return {key: native_result(item) for key, item in value.items()}
    if not callable(getattr(value, "to_host", None)):
        return value
    import sys
    torch = sys.modules.get("torch")
    if torch is None:
        import numpy as np
        from intent.runtime.storage_types import numpy_dtype
        return np.frombuffer(value.to_host(), dtype=numpy_dtype(value.dtype)).reshape(value.shape).copy()
    from intent.integrations.torch import torch_dtype
    from intent.language.dtypes import dtype

    element = torch_dtype(dtype(value.dtype))
    if not value.nbytes:
        return torch.empty(value.shape, dtype=element)
    # BF16 and FP8 NumPy extension dtypes cannot pass through from_numpy.
    # Interpret the public buffer's storage bytes without a numeric conversion.
    return torch.frombuffer(value.to_host(), dtype=element).reshape(value.shape).clone()
