from __future__ import annotations

import torch
import triton
import triton.language as tl


N = 32
BLOCK = (N, N)


@triton.jit
def _argmax_masked(values, mask):
    masked_values = tl.where(mask, values, -float("inf"))
    return tl.max(masked_values, axis=0, return_indices=True)


@triton.jit
def solve_symmetric_ldl_kernel(
    a_ptr,
    b_ptr,
    x_ptr,
    stride_a0,
    stride_a1,
    stride_b0,
    stride_b1,
    IS_HERMITIAN: tl.constexpr,
):
    row = tl.arange(0, N).to(tl.int32)
    col = tl.arange(0, N).to(tl.int32)
    matrix_index = row[:, None] * N + col[None, :]
    vector_index = tl.arange(0, N).to(tl.int32)

    raw = tl.load(a_ptr + row[:, None] * stride_a0 + col[None, :] * stride_a1)
    raw = raw.to(tl.float32)
    if IS_HERMITIAN:
        matrix = tl.where(row <= col, raw, tl.trans(raw))
    else:
        matrix = tl.where(row >= col, raw, tl.trans(raw))

    factors = tl.zeros(BLOCK, dtype=tl.float32)
    pivots = tl.zeros((N,), dtype=tl.float32)
    alpha = (1.0 + tl.sqrt(17.0)) / 8.0
    k = 0

    while k < N:
        active = row >= k
        diagonal = tl.sum(tl.where(row == col, matrix, 0.0), axis=0)
        diagonal_max, diagonal_index = _argmax_masked(diagonal, active)

        if k == N - 1:
            use_two = 0
            first_index = diagonal_index
        else:
            column = tl.where(active, tl.abs(matrix[row, k]), 0.0)
            column_max, candidate_index = _argmax_masked(column, active)
            use_two = 0
            first_index = diagonal_index
            if column_max > 0.0 and diagonal_max < alpha * column_max:
                candidate = tl.abs(matrix[candidate_index, k])
                candidate_row = tl.where(
                    row == candidate_index, 0.0, tl.abs(matrix[candidate_index, row])
                )
                candidate_row_max, _ = _argmax_masked(candidate_row, active)
                if candidate >= alpha * candidate_row_max:
                    first_index = candidate_index
                else:
                    use_two = 1

        second_index = first_index + 1
        if use_two != 0:
            second_index = k + 1

        if (use_two != 0 and first_index != k + 1) or (
            use_two == 0 and first_index != k
        ):
            row_source = tl.where(row == first_index, k, row)
            if use_two != 0:
                row_source = tl.where(row == second_index, k + 1, row_source)
                row_source = tl.where(row == k + 1, second_index, row_source)
            row_source = tl.where(row == k, first_index, row_source)
            matrix = tl.gather(matrix, row_source, axis=0)
            matrix = tl.gather(matrix, row_source, axis=1)

        if use_two != 0:
            b00 = matrix[k, k]
            b01 = matrix[k, k + 1]
            b11 = matrix[k + 1, k + 1]
            determinant = b00 * b11 - b01 * b01
            inv00 = b11 / determinant
            inv01 = -b01 / determinant
            inv11 = b00 / determinant

            future = row > k + 1
            block0 = tl.where(future, matrix[row, k], 0.0)
            block1 = tl.where(future, matrix[row, k + 1], 0.0)
            factor0 = inv00 * block0 + inv01 * block1
            factor1 = inv01 * block0 + inv11 * block1
            factors = tl.where((row == col) & (col == k), b00, factors)
            factors = tl.where(
                future[:, None] & (col[None, :] == k), factor0[:, None], factors
            )
            factors = tl.where(
                (row == col) & (col == k + 1), b11, factors
            )
            factors = tl.where(
                future[:, None] & (col[None, :] == k + 1), factor1[:, None], factors
            )

            update = (
                factor0[:, None] * factor0[None, :] * inv00
                + factor0[:, None] * factor1[None, :] * inv01
                + factor1[:, None] * factor0[None, :] * inv01
                + factor1[:, None] * factor1[None, :] * inv11
            )
            matrix -= tl.where(future[:, None] & future[None, :], update, 0.0)
            pivots = tl.where(row == k, -(k + 1), pivots)
            pivots = tl.where(row == k + 1, -(k + 2), pivots)
            k += 2
        else:
            pivot = matrix[k, k]
            future = row > k
            factor = tl.where(future, matrix[row, k] / pivot, 0.0)
            factors = tl.where((row == col) & (col == k), pivot, factors)
            factors = tl.where(
                future[:, None] & (col[None, :] == k), factor[:, None], factors
            )
            update = factor[:, None] * factor[None, :] * pivot
            matrix -= tl.where(future[:, None] & future[None, :], update, 0.0)
            pivots = tl.where(
                row == k,
                tl.where(pivot >= 0.0, k + 1, -(k + 1)),
                pivots,
            )
            k += 1

    scaled_factors = factors * pivots[None, :]
    packed_factors = tl.where(row < col, 0.0, factors)
    reconstructed = tl.dot(
        packed_factors,
        tl.trans(packed_factors),
        input_precision="ieee",
        out_dtype=tl.float32,
    )

    original_rhs = tl.load(
        b_ptr + vector_index * stride_b0 + tl.zeros((N,), dtype=tl.int32) * stride_b1
    )
    original_rhs = original_rhs.to(tl.float32)

    solve_matrix = reconstructed
    solve_rhs = original_rhs
    for pivot_row in range(0, N):
        below = row >= pivot_row
        candidates = tl.where(below, tl.abs(solve_matrix[row, pivot_row]), 0.0)
        _, selected = _argmax_masked(candidates, below)
        if selected != pivot_row:
            gather_index = tl.where(
                row == pivot_row,
                selected,
                tl.where(row == selected, pivot_row, row),
            )
            solve_matrix = tl.gather(solve_matrix, gather_index, axis=0)
            solve_rhs = tl.gather(solve_rhs, gather_index, axis=0)

        pivot_value = solve_matrix[pivot_row, pivot_row]
        below_pivot = row > pivot_row
        multipliers = tl.where(
            below_pivot, solve_matrix[row, pivot_row] / pivot_value, 0.0
        )
        solve_rhs -= multipliers * solve_rhs[pivot_row]
        solve_matrix -= multipliers[:, None] * solve_matrix[pivot_row, None, :]

    solution_scalar = 0.0
    solution = tl.zeros((N,), dtype=tl.float32)
    for reverse_index in range(0, N):
        solution_row = N - 1 - reverse_index
        previous_sum = tl.sum(
            tl.where(
                row > solution_row,
                solve_matrix[solution_row, row] * solution[row],
                0.0,
            ),
            axis=0,
        )
        solution_scalar = (
            solve_rhs[solution_row] - previous_sum
        ) / solve_matrix[solution_row, solution_row]
        solution = tl.where(row == solution_row, solution_scalar, solution)

    tl.store(
        x_ptr + vector_index * N,
        solution,
    )


@triton.jit
def copy_output_kernel(source_ptr, destination_ptr, stride_out0, stride_out1):
    row = tl.arange(0, N).to(tl.int32)
    values = tl.load(source_ptr + row * N)
    tl.store(destination_ptr + row * stride_out0 + stride_out1, values)


def build(context):
    state = {}

    def solve_symmetric_ldl(A, b, hermitian=False, out=None):
        device = A.device
        if state.get("device") != device:
            state["device"] = device
            state["x"] = torch.empty((N, 1), device=device, dtype=torch.float32)
        x = state["x"]
        solve_symmetric_ldl_kernel[(1,)](
            A,
            b,
            x,
            A.stride(0),
            A.stride(1),
            b.stride(0),
            b.stride(1),
            IS_HERMITIAN=bool(hermitian),
            num_warps=1,
            num_stages=1,
        )
        if out is None:
            return x

        copy_output_kernel[(1,)](
            x,
            out,
            out.stride(0),
            out.stride(1),
            num_warps=1,
            num_stages=1,
        )
        return out

    return solve_symmetric_ldl
