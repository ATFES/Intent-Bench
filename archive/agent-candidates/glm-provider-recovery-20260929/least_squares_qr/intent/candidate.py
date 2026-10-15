from __future__ import annotations

import torch

import intent
import intent.language as I


@intent.kernel
def _least_squares_qr_kernel(
    A: I.In[I.f32, ("M", "N")],
    b: I.In[I.f32, ("M",)],
    x: I.Out[I.f32, ("N",)],
) -> None:
    M, N = A.shape
    rows = I.domain(0, M)
    columns = I.domain(0, N)

    R = I.buffer((M, N), I.f32, init=A[rows, columns])
    transformed_b = I.buffer((M,), I.f32, init=b[rows])
    solution = I.buffer((N,), I.f32, init=I.zeros((N,), I.f32))
    two = I.cast(2.0, I.f32)

    for k in I.domain(0, N):
        active_rows = rows[k:M]
        column = R[active_rows, k]
        pivot_input = R[k, k]
        column_norm_squared = I.reduce.sum(
            column * column, axis=0, acc_dtype=I.f32
        )
        column_norm = I.sqrt(column_norm_squared)
        sign = I.select(
            pivot_input < 0.0,
            I.cast(-1.0, I.f32),
            I.cast(1.0, I.f32),
        )
        alpha = -sign * column_norm
        reflector = column - alpha
        reflector_norm_squared = I.reduce.sum(
            reflector * reflector, axis=0, acc_dtype=I.f32
        )
        beta = two / reflector_norm_squared
        reflector_column = I.reshape(
            reflector, (reflector.shape[0], 1)
        )

        trailing_columns = columns[k + 1 : N]
        trailing_block = R[active_rows, trailing_columns]
        row_projection = I.matmul(
            I.transpose(reflector_column),
            trailing_block,
            acc_dtype=I.f32,
        )
        R[active_rows, trailing_columns] = trailing_block - I.matmul(
            reflector_column,
            row_projection,
            acc_dtype=I.f32,
        ) * beta

        b_tail = transformed_b[active_rows]
        b_projection = I.dot(reflector, b_tail, acc_dtype=I.f32)
        transformed_b[active_rows] = b_tail - reflector * (beta * b_projection)

        R[k, k] = alpha
        R[active_rows, k] = I.cast(0.0, I.f32)

    for reverse_ordinal in I.domain(0, N):
        row = N - 1 - reverse_ordinal
        future_columns = columns[row + 1 : N]
        known_solution = solution[future_columns]
        future_coefficients = R[row, future_columns]
        known_sum = I.dot(
            future_coefficients, known_solution, acc_dtype=I.f32
        )
        pivot = R[row, row]
        solution[row] = (transformed_b[row] - known_sum) / pivot

    x[columns] = solution[columns]


def build(context):
    kernel = context.compile("least_squares_qr_householder", _least_squares_qr_kernel)

    def least_squares_qr(
        A: torch.Tensor,
        b: torch.Tensor,
        *,
        mode: str = "reduced",
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mode not in ("reduced", "complete"):
            raise ValueError("mode must be 'reduced' or 'complete'")
        if A.ndim != 2 or A.shape[0] < A.shape[1]:
            raise ValueError("A must be an overdetermined two-dimensional matrix")

        M, N = A.shape
        if b.ndim == 1:
            if b.shape[0] != M:
                raise ValueError("b has an incompatible shape")
            b_vector = b.view(M)
        elif b.ndim == 2 and b.shape[0] == M and b.shape[1] == 1:
            b_vector = b.view(M)
        else:
            raise ValueError("b must have shape (M,) or (M, 1)")

        if out is None:
            result = torch.empty((N, 1), dtype=A.dtype, device=A.device)
        else:
            result = out
            if result.shape != (N, 1):
                raise ValueError("out has an incompatible shape")

        kernel(A, b_vector, result.view(N))
        return result

    return least_squares_qr
