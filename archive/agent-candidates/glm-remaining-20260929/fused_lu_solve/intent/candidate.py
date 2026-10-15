from __future__ import annotations

import torch

import intent
import intent.language as I


@intent.kernel
def fused_lu_solve_kernel(
    A: I.In[I.f32, ("N", "N")],
    b: I.In[I.f32, ("N",)],
    x: I.Out[I.f32, ("N",)],
) -> None:
    size = A.shape[0]
    lu = I.buffer(A.shape, I.f32)
    triangular_rhs = I.buffer(b.shape, I.f32)

    for row in I.parallel(I.domain(0, size)):
        lu[row, :] = A[row, :]

    for pivot in range(size):
        column = lu[pivot:size, pivot]
        pivot_value, pivot_offset = I.arg_reduce.max(column, axis=0)
        pivot_index = pivot + pivot_offset

        pivot_row = lu[pivot_index, :]
        current_row = lu[pivot, :]
        for column_index in I.parallel(I.domain(0, size)):
            lu[pivot_index, column_index] = current_row[column_index]
            lu[pivot, column_index] = pivot_row[column_index]

        for row in I.parallel(I.domain(pivot + 1, size)):
            multiplier = lu[row, pivot] / pivot_value
            row_values = lu[row, :]
            lu[row, :] = row_values - multiplier * pivot_row

    for row in range(size):
        correction = I.reduce.sum(
            lu[row, 0:row] * triangular_rhs[0:row],
            axis=0,
        )
        triangular_rhs[row] = b[row] - correction

    for ordinal in range(size):
        row = size - 1 - ordinal
        suffix_start = row + 1
        correction = I.reduce.sum(
            lu[row, suffix_start:size]
            * triangular_rhs[suffix_start:size],
            axis=0,
        )
        x[row] = (triangular_rhs[row] - correction) / lu[row, row]


def build(context):
    kernel = context.compile(
        "fused_lu_solve_fixed_profile",
        fused_lu_solve_kernel,
    )

    def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        result = torch.empty(
            b.shape,
            dtype=A.dtype,
            device=b.device,
        )
        kernel(A, b, result)
        return result

    return fused_lu_solve
