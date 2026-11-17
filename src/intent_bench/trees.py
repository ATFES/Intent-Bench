"""Copies and comparisons that preserve array views and shared mutable storage."""
from __future__ import annotations

import copy
from dataclasses import fields, is_dataclass
import numpy as np


def clone(value, memo=None):
    memo = {} if memo is None else memo
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, np.ndarray):
        owner = value
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        if not (owner.flags.c_contiguous or owner.flags.f_contiguous):
            raise ValueError("array owner must have contiguous storage; supply a custom input container")
        if id(owner) not in memo:
            memo[id(owner)] = owner.copy(order="K")
        copied = memo[id(owner)]
        if value is not owner:
            offset = value.ctypes.data - owner.ctypes.data
            copied = np.ndarray(value.shape, dtype=value.dtype, buffer=copied,
                                offset=offset, strides=value.strides)
        memo[id(value)] = copied
        return copied
    if isinstance(value, dict):
        result = {}
        memo[id(value)] = result
        result.update((k, clone(v, memo)) for k, v in value.items())
        return result
    if isinstance(value, (list, tuple)):
        result = [clone(v, memo) for v in value]
        result = tuple(result) if isinstance(value, tuple) else result
        memo[id(value)] = result
        return result
    if is_dataclass(value) and not isinstance(value, type):
        result = type(value)(**{field.name: clone(getattr(value, field.name), memo)
                               for field in fields(value) if field.init})
        memo[id(value)] = result
        return result
    return copy.deepcopy(value, memo)


def restore(destination, source):
    if isinstance(destination, np.ndarray):
        np.copyto(destination, source)
    elif isinstance(destination, dict):
        for key in destination:
            restore(destination[key], source[key])
    elif isinstance(destination, (list, tuple)):
        for a, b in zip(destination, source, strict=True):
            restore(a, b)
    elif hasattr(destination, "copy_"):
        destination.copy_(source)


def to_device(value, device):
    """Move a tensor tree once, preserving repeated objects and storage aliases."""
    import torch
    tensors, allocations = {}, {}
    def move(item):
        if isinstance(item, torch.Tensor):
            if id(item) in tensors:
                return tensors[id(item)]
            if item.layout != torch.strided or item.is_conj() or item.is_neg():
                raise NotImplementedError("input transfer requires ordinary strided tensor storage")
            storage = item.untyped_storage()
            key = (str(item.device), storage.data_ptr(), storage.nbytes())
            if key not in allocations:
                raw = torch.empty(0, dtype=torch.uint8, device=item.device).set_(
                    storage, 0, (storage.nbytes(),), (1,))
                allocations[key] = raw.to(device)
            result = allocations[key].view(item.dtype).as_strided(item.shape, item.stride(), item.storage_offset())
            tensors[id(item)] = result
            return result
        if isinstance(item, tuple):
            return tuple(move(v) for v in item)
        if isinstance(item, list):
            return [move(v) for v in item]
        if isinstance(item, dict):
            return {key: move(v) for key, v in item.items()}
        return item
    return move(value)


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        # NumPy does not understand bf16 or fp8 storage.
        if str(value.dtype) in {"torch.bfloat16", "torch.float8_e4m3fn", "torch.float8_e5m2"}:
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def assert_close(actual, expected, *, atol, rtol):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError("output dictionary structure differs")
        for key in expected:
            assert_close(actual[key], expected[key], atol=atol, rtol=rtol)
    elif isinstance(expected, (tuple, list)):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise AssertionError("output sequence structure differs")
        for a, b in zip(actual, expected, strict=True):
            assert_close(a, b, atol=atol, rtol=rtol)
    else:
        a, b = array(actual), array(expected)
        if a.shape != b.shape:
            raise AssertionError(f"output shapes differ: {a.shape} vs {b.shape}")
        if a.dtype.kind in "biu" and b.dtype.kind in "biu":
            np.testing.assert_array_equal(a, b)
        else:
            np.testing.assert_allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)
