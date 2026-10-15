from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def lower_symmetric_power_two(
    matrix: I.In[I.f32, ("M", "N"), I.constraints(noalias=True)],
    result: I.Out[I.f32, ("M", "N"), I.constraints(noalias=True)],
) -> None:
    rows = I.domain(0, matrix.shape[0])
    columns = I.domain(0, matrix.shape[1])

    row_indices = I.indices(rows)
    column_indices = I.indices(columns)
    row_indices_broad = I.reshape(row_indices, (row_indices.shape[0], 1))
    column_indices_broad = I.reshape(column_indices, (1, column_indices.shape[0]))
    lower_or_diagonal = row_indices_broad >= column_indices_broad

    values = matrix[rows, columns]
    transposed_values = I.transpose(values)
    symmetric_values = I.select(lower_or_diagonal, values, transposed_values)
    product = I.matmul(
        symmetric_values,
        symmetric_values,
        acc_dtype=I.f32,
        transpose_lhs=True,
        transpose_rhs=False,
    )
    result[rows, columns] = product


@intent.kernel
def symmetrize_lower_triangle(
    matrix: I.In[I.f32, ("M", "N"), I.constraints(noalias=True)],
    symmetric: I.Out[I.f32, ("M", "N"), I.constraints(noalias=True)],
) -> None:
    rows = I.domain(0, matrix.shape[0])
    columns = I.domain(0, matrix.shape[1])

    row_indices = I.indices(rows)
    column_indices = I.indices(columns)
    row_indices_broad = I.reshape(row_indices, (row_indices.shape[0], 1))
    column_indices_broad = I.reshape(column_indices, (1, column_indices.shape[0]))
    lower_or_diagonal = row_indices_broad >= column_indices_broad

    values = matrix[rows, columns]
    transposed_values = I.transpose(values)
    symmetric_values = I.select(lower_or_diagonal, values, transposed_values)
    symmetric[rows, columns] = symmetric_values


@intent.kernel
def symmetric_square(
    symmetric: I.In[I.f32, ("M", "N"), I.constraints(noalias=True)],
    result: I.Out[I.f32, ("M", "N"), I.constraints(noalias=True)],
) -> None:
    rows = I.domain(0, symmetric.shape[0])
    columns = I.domain(0, symmetric.shape[1])
    product = I.matmul(
        symmetric,
        symmetric,
        acc_dtype=I.f32,
        transpose_lhs=True,
        transpose_rhs=False,
    )
    result[rows, columns] = product


def build(context):
    power_two = context.compile("matrix_power_eig_power_two", lower_symmetric_power_two)
    symmetrize = context.compile("matrix_power_eig_symmetrize", symmetrize_lower_triangle)
    square = context.compile("matrix_power_eig_square", symmetric_square)

    _workspace: torch.Tensor | None = None

    def matrix_power_eig(
        A: torch.Tensor,
        k: float,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nonlocal _workspace
        supplied_out = out is not None
        if out is None:
            out = torch.empty_like(A)

        if supplied_out and _workspace is None:
            _workspace = torch.empty_like(A)

        if supplied_out:
            symmetrize(A, _workspace)
            square(_workspace, out)
        else:
            power_two(A, out)
        return out

    return matrix_power_eig
