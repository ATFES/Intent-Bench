from __future__ import annotations

from typing import List, Tuple, Union

import intent
import intent.language as I
import torch


@intent.kernel
def tensordot_kernel(
    a: I.In[I.f16, ("M", "K0", "K1")],
    b: I.In[I.f16, ("K0", "K1", "N")],
    output: I.Out[I.f16, ("M", "N")],
) -> None:
    contracted = I.contract(
        a,
        b,
        reduce=((1, 0), (2, 1)),
        batch=(),
        acc_dtype=I.f32,
    )
    output[:, :] = I.cast(contracted, I.f16)


def _normalise_dims(
    a: torch.Tensor,
    b: torch.Tensor,
    dims: Union[int, Tuple[List[int], List[int]], List[List[int]]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if isinstance(dims, int):
        if dims < 0:
            raise ValueError("dims must be non-negative")
        if dims > a.ndim or dims > b.ndim:
            raise ValueError("contraction rank exceeds tensor rank")
        return (
            tuple(range(a.ndim - dims, a.ndim)),
            tuple(range(dims)),
        )

    if not isinstance(dims, (tuple, list)) or len(dims) != 2:
        raise ValueError("dims must be an int or a pair of dimension lists")

    dims_a, dims_b = dims
    if not isinstance(dims_a, (tuple, list)) or not isinstance(dims_b, (tuple, list)):
        raise ValueError("dims must contain two dimension lists")
    if len(dims_a) != len(dims_b):
        raise ValueError("contraction dimension lists must have equal length")

    axes_a = tuple(int(dim) % a.ndim for dim in dims_a)
    axes_b = tuple(int(dim) % b.ndim for dim in dims_b)
    if len(set(axes_a)) != len(axes_a) or len(set(axes_b)) != len(axes_b):
        raise ValueError("contraction dimensions must be unique")
    return axes_a, axes_b


def build(context):
    tensordot = context.compile("tensordot_generic_matrix_product", tensordot_kernel)

    def tensordot_wrapper(
        a: torch.Tensor,
        b: torch.Tensor,
        dims: Union[int, Tuple[List[int], List[int]], List[List[int]]] = 2,
    ) -> torch.Tensor:
        axes_a, axes_b = _normalise_dims(a, b, dims)
        if a.ndim != 3 or b.ndim != 3:
            raise ValueError("the compiled profile requires rank-3 inputs")
        if axes_a != (1, 2) or axes_b != (0, 1):
            raise ValueError("the compiled profile requires contraction axes (1, 2) and (0, 1)")
        if a.shape[1:3] != b.shape[0:2]:
            raise ValueError("contraction dimensions must have matching sizes")

        output = torch.empty(
            (a.shape[0], b.shape[2]),
            device=a.device,
            dtype=a.dtype,
        )
        tensordot(a, b, output)
        return output

    return tensordot_wrapper
