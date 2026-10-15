from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def lu_factor(
    A: I.In[I.f32, (256, 256)],
    LU: I.Out[I.f32, (256, 256)],
):
    rows = I.domain(0, 256)
    columns = I.domain(0, 256)
    lu = I.buffer((256, 256), I.f32, init=A[rows, columns])

    # Each pivot step is ordered; rows below the pivot are independent.
    for k in range(256):
        tail = I.domain(k, 256)
        _, pivot_offset = I.arg_reduce.max(I.abs(lu[tail, k]), axis=0)
        pivot = k + pivot_offset

        for j in I.parallel(columns):
            top = lu[k, j]
            selected = lu[pivot, j]
            lu[k, j] = selected
            lu[pivot, j] = top

        diagonal = lu[k, k]
        for i in I.parallel(I.domain(k + 1, 256)):
            multiplier = lu[i, k] / diagonal
            lu[i, k] = multiplier
            for j in I.parallel(I.domain(k + 1, 256)):
                lu[i, j] = lu[i, j] - multiplier * lu[k, j]

    LU[rows, columns] = lu[rows, columns]


@intent.kernel
def lu_solve(
    LU: I.In[I.f32, (256, 256)],
    b: I.In[I.f32, (256,)],
    x: I.Out[I.f32, (256,)],
):
    y = I.buffer((256,), I.f32)

    # Forward substitution for the unit lower factor.
    for i in range(256):
        prior = I.domain(0, i)
        contribution = I.reduce.sum(LU[i, prior] * y[prior], axis=0, acc_dtype=I.f32)
        y[i] = b[i] - contribution

    # Reverse substitution for the upper factor.
    for ordinal in range(256):
        i = 256 - 1 - ordinal
        following = I.domain(i + 1, 256)
        contribution = I.reduce.sum(LU[i, following] * x[following], axis=0, acc_dtype=I.f32)
        x[i] = (y[i] - contribution) / LU[i, i]


def build(context):
    factor_artifact = context.compile("lu_factor", lu_factor)
    solve_artifact = context.compile("lu_solve", lu_solve)

    def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        lu = torch.empty_like(A)
        output = torch.empty_like(b)
        factor_artifact(A, lu)
        solve_artifact(lu, b, output)
        return output

    return fused_lu_solve
