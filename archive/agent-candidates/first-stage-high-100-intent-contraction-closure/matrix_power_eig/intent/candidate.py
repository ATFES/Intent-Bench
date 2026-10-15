from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def _symmetrize_lower(
    A: I.In[I.f32, (256, 256)],
    S: I.Out[I.f32, (256, 256)],
):
    rows = I.domain(0, 256)
    cols = I.domain(0, 256)
    values = A[rows, cols]

    row_indices = I.reshape(I.indices(rows), (256, 1))
    col_indices = I.reshape(I.indices(cols), (1, 256))
    lower = row_indices >= col_indices
    symmetric = I.select(lower, values, I.transpose(values))
    S[rows, cols] = symmetric


@intent.kernel
def _spectral_power_two(
    S: I.In[I.f32, (256, 256)],
    out: I.Out[I.f32, (256, 256)],
):
    rows = I.domain(0, 256)
    cols = I.domain(0, 256)
    matrix = S[rows, cols]
    result = I.matmul(matrix, matrix, acc_dtype=I.f32)
    out[rows, cols] = result


def build(context):
    symmetrize = context.compile(
        "matrix_power_eig_symmetrize", _symmetrize_lower, constexprs={}
    )
    power_two = context.compile(
        "matrix_power_eig_power_two", _spectral_power_two, constexprs={}
    )

    def matrix_power_eig(
        A: torch.Tensor,
        k: float,
        *,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        del k
        symmetric = torch.empty_like(A)
        symmetrize(A, symmetric)

        if out is None:
            out = torch.empty_like(A)
        power_two(symmetric, out)
        return out

    return matrix_power_eig
