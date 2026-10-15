from __future__ import annotations

from typing import Callable

import intent
import intent.language as I


@intent.kernel
def _sigmoid_argmax_dim0(
    values: I.In[I.f32, ("M", "N"), I.constraints(noalias=True)],
    indices: I.Out[I.i64, ("N",), I.constraints(noalias=True)],
) -> None:
    probabilities = I.sigmoid(values[:, :])
    _, selected = I.arg_reduce.max(probabilities, axis=0)
    indices[:] = I.cast(selected, I.i64)


@intent.kernel
def _sigmoid_argmax_dim1(
    values: I.In[I.f32, ("M", "N"), I.constraints(noalias=True)],
    indices: I.Out[I.i64, ("M",), I.constraints(noalias=True)],
) -> None:
    probabilities = I.sigmoid(values[:, :])
    _, selected = I.arg_reduce.max(probabilities, axis=1)
    indices[:] = I.cast(selected, I.i64)


@intent.kernel
def _sigmoid_argmax_all(
    values: I.In[I.f32, ("K",), I.constraints(noalias=True)],
    indices: I.Out[I.i64, (), I.constraints(noalias=True)],
) -> None:
    probabilities = I.sigmoid(values[:])
    _, selected = I.arg_reduce.max(probabilities, axis=0)
    indices[()] = I.cast(selected, I.i64)


def build(context) -> Callable:
    reduce_dim0 = context.compile("sigmoid_argmax_dim0", _sigmoid_argmax_dim0)
    reduce_dim1 = context.compile("sigmoid_argmax_dim1", _sigmoid_argmax_dim1)
    reduce_all = context.compile("sigmoid_argmax_all", _sigmoid_argmax_all)

    def sigmoid_argmax(input, dim=None, keepdim=False):
        if input.ndim != 2:
            raise ValueError("sigmoid_argmax expects a two-dimensional input")

        if dim is None:
            result = reduce_all.run(input.reshape(-1))
            return result

        normalized_dim = dim
        if normalized_dim < 0:
            normalized_dim += input.ndim
        if normalized_dim not in (0, 1):
            raise ValueError("Dimension out of range")

        if normalized_dim == 0:
            result = reduce_dim0.run(input)
            if keepdim:
                return result.unsqueeze(0)
            return result

        result = reduce_dim1.run(input)
        if keepdim:
            return result.unsqueeze(1)
        return result

    return sigmoid_argmax
