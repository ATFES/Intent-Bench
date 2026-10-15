from __future__ import annotations

import intent
import intent.language as I
import torch


BLOCK_SIZE = 32


@intent.kernel
def copy_augmented(
    A: I.In[I.f32, ("M", "N")],
    B: I.In[I.f32, ("M", "R")],
    work: I.Out[I.f32, ("M", "W")],
) -> None:
    matrix_rows = I.domain(0, A.shape[0])
    matrix_columns = I.domain(0, A.shape[1])
    rhs_columns = I.domain(0, B.shape[1])

    for row in I.parallel(matrix_rows):
        for column in I.parallel(matrix_columns):
            work[row, column] = A[row, column]
        for column in I.parallel(rhs_columns):
            work[row, A.shape[1] + column] = B[row, column]


@intent.kernel
def panel_lu(
    work: I.InOut[I.f32, ("M", "W")],
    panel_begin: I.i64,
    panel_end: I.i64,
) -> None:
    width = work.shape[1]

    for pivot in I.domain(panel_begin, panel_end):
        pivot_column = work[I.domain(pivot, work.shape[0]), pivot]
        _, pivot_local = I.arg_reduce.max(I.abs(pivot_column), axis=0)
        pivot_row = pivot + pivot_local

        all_columns = I.domain(0, width)
        for column in I.parallel(all_columns):
            current_pivot = work[pivot, column]
            current_other = work[pivot_row, column]
            work[pivot, column] = current_other
            work[pivot_row, column] = current_pivot

        pivot_value = work[pivot, pivot]
        update_columns = I.domain(pivot + 1, width)
        pivot_row_values = work[pivot, update_columns]

        for row in I.parallel(I.domain(pivot + 1, work.shape[0])):
            factor = work[row, pivot] / pivot_value
            work[row, pivot] = factor
            row_values = work[row, update_columns]
            work[row, update_columns] = row_values - factor * pivot_row_values


@intent.kernel
def trailing_update(
    work: I.InOut[I.f32, ("M", "W")],
    panel_begin: I.i64,
    panel_end: I.i64,
) -> None:
    panel_rows = I.domain(panel_begin, panel_end)
    trailing_rows = I.domain(panel_end, work.shape[0])
    trailing_columns = I.domain(panel_end, work.shape[1])
    panel_values = work[trailing_rows, panel_rows]
    trailing_values = work[panel_rows, trailing_columns]
    update = I.matmul(panel_values, trailing_values, acc_dtype=I.f32)
    work[trailing_rows, trailing_columns] = (
        work[trailing_rows, trailing_columns] - update
    )


@intent.kernel
def triangular_solve(
    work: I.In[I.f32, ("M", "W")],
    X: I.Out[I.f32, ("M", "R")],
) -> None:
    row_count = work.shape[0]
    rhs_count = X.shape[1]
    solution = I.buffer((row_count, rhs_count), I.f32)
    rhs_axes = I.domain(0, rhs_count)

    for row in I.domain(0, row_count):
        values = work[row, rhs_axes]
        if row > 0:
            prefix = I.domain(0, row)
            coefficients = work[row, prefix]
            prior = solution[prefix, rhs_axes]
            correction = I.matmul(
                I.reshape(coefficients, (1, coefficients.shape[0])),
                prior,
                acc_dtype=I.f32,
            )
            values = values - I.reshape(correction, (rhs_count,))
        solution[row, rhs_axes] = values / work[row, row]

    for ordinal in I.domain(0, row_count):
        row = row_count - 1 - ordinal
        values = solution[row, rhs_axes]
        if row + 1 < row_count:
            suffix = I.domain(row + 1, row_count)
            coefficients = work[row, suffix]
            later = solution[suffix, rhs_axes]
            correction = I.matmul(
                I.reshape(coefficients, (1, coefficients.shape[0])),
                later,
                acc_dtype=I.f32,
            )
            values = values - I.reshape(correction, (rhs_count,))
        solution[row, rhs_axes] = values / work[row, row]

    rows = I.domain(0, row_count)
    for row in I.parallel(rows):
        for rhs in I.parallel(rhs_axes):
            X[row, rhs] = solution[row, rhs]


def _solve_matrix(A, B, out, artifacts):
    copy_artifact, panel_artifact, trailing_artifact, solve_artifact = artifacts
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("this specialization expects rank-2 A and B")
    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be square")
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have the same number of rows")
    if A.dtype != B.dtype:
        raise ValueError("A and B must have the same dtype")

    row_count = A.shape[0]
    rhs_count = B.shape[1]
    work = torch.empty(
        (row_count, row_count + rhs_count),
        dtype=A.dtype,
        device=A.device,
    )
    if out is None:
        result = torch.empty(
            (row_count, rhs_count), dtype=A.dtype, device=A.device
        )
    else:
        if (
            out.shape != (row_count, rhs_count)
            or out.dtype != A.dtype
            or out.device != A.device
        ):
            raise ValueError("out has an incompatible shape, dtype, or device")
        result = out

    copy_artifact(A, B, work)
    for panel_begin in range(0, row_count, BLOCK_SIZE):
        panel_end = min(panel_begin + BLOCK_SIZE, row_count)
        panel_artifact(work, panel_begin, panel_end)
        if panel_end < row_count:
            trailing_artifact(work, panel_begin, panel_end)
    solve_artifact(work, result)
    torch.cuda.synchronize(device=A.device)
    return result


def build(context):
    copy_artifact = context.compile("solve_copy_augmented_f32", copy_augmented)
    panel_artifact = context.compile("solve_panel_lu_f32", panel_lu)
    trailing_artifact = context.compile("solve_trailing_update_f32", trailing_update)
    solve_artifact = context.compile("solve_triangular_f32", triangular_solve)

    def solve(A, B, *, left=True, out=None):
        if A.device.type != "cuda" or B.device.type != "cuda":
            raise ValueError("A and B must be CUDA tensors")
        if A.dtype != torch.float32:
            raise ValueError("this specialization supports float32 inputs")

        b_matrix = B.reshape((B.shape[0], 1)) if B.ndim == 1 else B
        if left:
            a_used = A
            b_used = b_matrix
            out_used = out
            if out is not None and B.ndim == 1:
                out_used = out.reshape((B.shape[0], 1))
            result = _solve_matrix(
                a_used,
                b_used,
                out_used,
                (copy_artifact, panel_artifact, trailing_artifact, solve_artifact),
            )
            if B.ndim == 1 and out is None:
                return result.reshape(B.shape)
            return result

        a_used = A.T
        b_used = b_matrix if B.ndim == 1 else b_matrix.T
        out_used = None if out is None else out.T
        if out is not None and B.ndim == 1:
            out_used = out.reshape((B.shape[0], 1))
        transposed_result = _solve_matrix(
                a_used,
                b_used,
                out_used,
                (copy_artifact, panel_artifact, trailing_artifact, solve_artifact),
            )
        if out is not None:
            return out
        if B.ndim == 1:
            return transposed_result.reshape(B.shape)
        return transposed_result.T

    return solve
