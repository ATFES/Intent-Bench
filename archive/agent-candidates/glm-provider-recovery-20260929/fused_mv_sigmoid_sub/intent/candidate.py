from __future__ import annotations

from typing import Any, Callable

import intent
import intent.language as I
import torch


@intent.kernel
def _matrix_vector(
    matrix: I.In[I.f32, ("N", "K")],
    vector: I.In[I.f32, ("K",)],
    product: I.Out[I.f32, ("N",)],
):
    for row in I.parallel(I.domain(0, matrix.shape[0])):
        terms = matrix[row, :] * vector
        product[row] = I.reduce.sum(terms, axis=0, acc_dtype=I.f32)


@intent.kernel
def _sigmoid_subtract_scalar(
    product: I.In[I.f32, ("N",)],
    result: I.Out[I.f32, ("N",)],
    offset: I.f32,
):
    for row in I.parallel(I.domain(0, product.shape[0])):
        result[row] = I.sigmoid(product[row]) - offset


@intent.kernel
def _sigmoid_subtract_tensor(
    product: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    result: I.Out[I.f32, ("N",)],
    alpha: I.f32,
):
    for row in I.parallel(I.domain(0, product.shape[0])):
        result[row] = I.sigmoid(product[row]) - alpha * other[row]


def _as_broadcast_vector(value: Any, length: int) -> Any:
    if value.ndim == 0:
        return value.view(1).expand(length)
    if value.ndim == 1 and value.shape[0] in (1, length):
        return value.expand(length)
    raise ValueError("other must be broadcastable to the output shape")


def build(context: Any) -> Callable[..., Any]:
    matrix_vector = context.compile(
        "fused_mv_sigmoid_sub_matrix_vector", _matrix_vector
    )
    sigmoid_subtract_scalar = context.compile(
        "fused_mv_sigmoid_sub_scalar", _sigmoid_subtract_scalar
    )
    sigmoid_subtract_tensor = context.compile(
        "fused_mv_sigmoid_sub_tensor", _sigmoid_subtract_tensor
    )

    def fused_mv_sigmoid_sub(
        input: Any,
        vec: Any,
        other: Any,
        alpha: Any = 1,
        *,
        out: Any = None,
    ) -> Any:
        rows = input.shape[0]
        product = torch.empty((rows,), dtype=input.dtype, device=input.device)
        if out is None:
            result = torch.empty(
                (rows,), dtype=input.dtype, device=input.device
            )
        else:
            result = out

        matrix_vector(input, vec, product)
        if isinstance(other, torch.Tensor):
            broadcast_other = _as_broadcast_vector(other, rows)
            sigmoid_subtract_tensor(
                product, broadcast_other, result, float(alpha)
            )
        else:
            sigmoid_subtract_scalar(
                product, result, float(alpha) * float(other)
            )
        return result

    return fused_mv_sigmoid_sub
