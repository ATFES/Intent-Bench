from __future__ import annotations

import torch

import intent
import intent.language as I


@intent.kernel
def fused_householder_qr(
    A: I.In[I.f32, ("M", "N")],
    R: I.Out[I.f32, ("M", "N")],
    B: I.In[I.f32, ("M", "K")],
    QT_b: I.Out[I.f32, ("M", "K")],
) -> None:
    M, N = A.shape
    K = B.shape[1]
    rows = I.domain(0, M)
    columns = I.domain(0, N)
    rhs = I.domain(0, K)
    row_coordinates = I.indices(rows)
    column_coordinates = I.indices(columns)

    R[:, :] = A[:, :]
    QT_b[:, :] = B[:, :]

    for step in I.domain(0, N):
        active_rows = row_coordinates >= step
        active_columns = column_coordinates >= step
        current_column = R[:, step]
        column_energy = I.reduce.sum(
            I.select(active_rows, current_column * current_column, 0.0),
            axis=0,
        )
        diagonal = R[step, step]
        column_norm = I.sqrt(column_energy)
        sign = I.cast(I.select(diagonal >= 0.0, 1.0, -1.0), I.f32)
        shifted_diagonal = diagonal + sign * column_norm
        reflector_energy = column_energy + shifted_diagonal * shifted_diagonal
        beta = I.cast(2.0, I.f32) / reflector_energy

        reflector = I.full((M,), 0.0, I.f32)
        reflector[:, step] = I.select(
            active_rows,
            current_column,
            0.0,
        )
        reflector[step] = shifted_diagonal

        for row in I.parallel(rows):
            if active_rows[row]:
                scale = beta * reflector[row]
                old_row = R[row, columns]
                R[row, columns] = I.select(
                    active_columns,
                    old_row + scale * reflector[row],
                    old_row,
                )

                rhs_row = QT_b[row, rhs]
                QT_b[row, rhs] = rhs_row + scale * rhs_row


@intent.kernel
def triangular_solve(
    R: I.In[I.f32, ("M", "N")],
    QT_b: I.In[I.f32, ("M", "K")],
    X: I.Out[I.f32, ("N", "K")],
) -> None:
    N = R.shape[1]
    K = QT_b.shape[1]
    columns = I.domain(0, N)
    rhs = I.domain(0, K)
    column_coordinates = I.indices(columns)
    X[:, :] = I.full(X.shape, 0.0, I.f32)

    for reverse_step in I.domain(0, N):
        row = N - 1 - reverse_step
        remaining = column_coordinates > row
        R_row = R[row, columns]
        current_X = X[columns, rhs]
        residual_sum = I.reduce.sum(
            I.select(
                remaining,
                R_row * current_X,
                0.0,
            ),
            axis=0,
        )
        X[row, rhs] = (QT_b[row, rhs] - residual_sum) / R[row, row]


def build(context):
    qr_kernel = context.compile(
        "fused_qr_solve_householder",
        fused_householder_qr,
    )
    solve_kernel = context.compile(
        "fused_qr_solve_back_substitution",
        triangular_solve,
    )

    def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        R = torch.empty_like(A)
        QT_b = torch.empty_like(b)
        X = torch.empty((A.shape[1], b.shape[1]), dtype=A.dtype, device=A.device)
        qr_kernel(A, R, b, QT_b)
        solve_kernel(R, QT_b, X)
        return X

    return fused_qr_solve
