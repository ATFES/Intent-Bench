from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _lu_solve_kernel(
    a_ptr,
    b_ptr,
    x_ptr,
    n,
    k,
    batch_count,
    a_row_stride,
    a_col_stride,
    b_row_stride,
    b_col_stride,
    x_row_stride,
    x_col_stride,
    N: tl.constexpr,
    K: tl.constexpr,
    N_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
    EXACT: tl.constexpr,
    PIVOT: tl.constexpr,
):
    batch = tl.program_id(0)
    rows = tl.arange(0, N_BLOCK)
    cols = tl.arange(0, K_BLOCK)
    batch_offset = batch.to(tl.int64)

    if EXACT:
        a_offsets = (
            batch_offset * n * n
            + rows[:, None] * a_row_stride
            + cols[None, :] * a_col_stride
        )
        matrix = tl.load(a_ptr + a_offsets)
        b_offsets = (
            batch_offset * n * k
            + rows[:, None] * b_row_stride
            + cols[None, :] * b_col_stride
        )
        rhs = tl.load(b_ptr + b_offsets)
    else:
        a_offsets = (
            batch_offset * n * n
            + rows[:, None] * a_row_stride
            + cols[None, :] * a_col_stride
        )
        a_mask = (rows[:, None] < n) & (rows[None, :] < n)
        matrix = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0)
        b_offsets = (
            batch_offset * n * k
            + rows[:, None] * b_row_stride
            + cols[None, :] * b_col_stride
        )
        b_mask = (rows[:, None] < n) & (cols[None, :] < k)
        rhs = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0)

    for j in range(0, n):
        column = tl.sum(tl.where(rows[:, None] == j, matrix, 0.0), axis=0)
        if PIVOT:
            pivot_candidates = tl.where(rows >= j, tl.abs(column), -1.0)
            pivot_row = tl.argmax(pivot_candidates, axis=0, tie_break_left=True)
            pivot_row = pivot_row.to(tl.int32)
            source_rows = tl.where(
                rows == j,
                pivot_row,
                tl.where(rows == pivot_row, j, rows),
            )
            matrix_index = source_rows[:, None] + 0 * rows[None, :]
            rhs_index = source_rows[:, None] + 0 * cols[None, :]
            matrix = tl.gather(matrix, matrix_index, axis=0)
            rhs = tl.gather(rhs, rhs_index, axis=0)
            column = tl.sum(tl.where(rows[:, None] == j, matrix, 0.0), axis=0)

        pivot_value = tl.sum(tl.where(rows == j, column, 0.0), axis=0)
        column = column / pivot_value
        active_row = tl.sum(tl.where(rows[:, None] == j, matrix, 0.0), axis=0)
        active_rhs = tl.sum(tl.where(rows[:, None] == j, rhs, 0.0), axis=0)
        lower = rows > j
        matrix = tl.where(
            lower[:, None],
            matrix - column[:, None] * active_row[None, :],
            matrix,
        )
        rhs = tl.where(
            lower[:, None],
            rhs - column[:, None] * active_rhs[None, :],
            rhs,
        )

    for j_index in range(0, n):
        j = n - 1 - j_index
        factor_row = tl.sum(tl.where(rows[:, None] == j, matrix, 0.0), axis=0)
        diagonal = tl.sum(tl.where(rows == j, factor_row, 0.0), axis=0)
        solved_row = tl.sum(tl.where(rows[:, None] == j, rhs, 0.0), axis=0)
        correction = tl.sum(factor_row[:, None] * rhs, axis=0)
        solved_row = (solved_row - correction) / diagonal
        rhs = tl.where(rows[:, None] == j, solved_row[None, :], rhs)

    if EXACT:
        x_offsets = (
            batch_offset * n * k
            + rows[:, None] * x_row_stride
            + cols[None, :] * x_col_stride
        )
        tl.store(x_ptr + x_offsets, rhs)
    else:
        x_offsets = (
            batch_offset * n * k
            + rows[:, None] * x_row_stride
            + cols[None, :] * x_col_stride
        )
        x_mask = (rows[:, None] < n) & (cols[None, :] < k)
        tl.store(x_ptr + x_offsets, rhs, mask=x_mask)


def build(context):
    def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
        if A.ndim < 2 or Bs.ndim != A.ndim:
            raise ValueError("A and Bs must have matching batch dimensions")

        n = A.shape[-2]
        k = Bs.shape[-1]
        if A.shape[-1] != n or Bs.shape[-2] != n:
            raise ValueError("A and Bs have incompatible matrix dimensions")
        if A.dtype != torch.float32 or Bs.dtype != torch.float32:
            raise ValueError("A and Bs must be float32 tensors")
        if not A.is_cuda or not Bs.is_cuda:
            raise ValueError("A and Bs must be CUDA tensors")

        if n == 64 and k == 64:
            n_block = 64
            k_block = 64
            exact = True
        else:
            n_block = max(1, triton.next_power_of_2(n))
            k_block = max(1, triton.next_power_of_2(k))
            exact = n == n_block and k == k_block

        if n_block > 128 or k_block > 256:
            raise ValueError("Matrix or RHS dimensions exceed the supported tile size")

        if out is None:
            output = torch.empty_like(Bs)
        else:
            if (
                out.shape != Bs.shape
                or out.dtype != Bs.dtype
                or out.device != Bs.device
                or not out.is_contiguous()
            ):
                raise ValueError("out must be contiguous with the shape and dtype of Bs")
            output = out

        if A.numel() == 0:
            return output
        if not A.is_contiguous() or not Bs.is_contiguous():
            raise ValueError("A and Bs must be contiguous")

        batch_count = A.numel() // (n * n)
        _lu_solve_kernel[(batch_count,)](
            A,
            Bs,
            output,
            n,
            k,
            batch_count,
            A.stride(-2),
            A.stride(-1),
            Bs.stride(-2),
            Bs.stride(-1),
            output.stride(-2),
            output.stride(-1),
            N=n,
            K=k,
            N_BLOCK=n_block,
            K_BLOCK=k_block,
            EXACT=exact,
            PIVOT=pivot,
            num_warps=8,
        )
        return output

    return solve_multiple_lu
