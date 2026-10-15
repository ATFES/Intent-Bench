from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_lu_solve_kernel(
    a_ptr,
    b_ptr,
    work_ptr,
    x_ptr,
    n,
    stride_a_row,
    stride_a_col,
    stride_b,
    stride_x,
    BLOCK_N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    cols = tl.arange(0, BLOCK_N)
    row_ids = tl.arange(0, BLOCK_ROWS)
    col_mask = cols < n

    for row_start in range(0, n, BLOCK_ROWS):
        rows = row_start + row_ids[:, None]
        offsets = rows * stride_a_row + cols[None, :] * stride_a_col
        mask = (rows < n) & col_mask[None, :]
        values = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        tl.store(work_ptr + offsets, values, mask=mask)

    for k in range(0, n):
        pivot_rows = k + row_ids
        column_offsets = pivot_rows * stride_a_row + k * stride_a_col
        column_mask = pivot_rows < n
        column = tl.load(work_ptr + column_offsets, mask=column_mask, other=-1.0)
        pivot_local = tl.argmax(tl.abs(column), axis=0, tie_break_left=True)
        pivot = k + pivot_local.to(tl.int32)

        pivot_row_offsets = k * stride_a_row + cols * stride_a_col
        selected_row_offsets = pivot * stride_a_row + cols * stride_a_col
        pivot_row = tl.load(work_ptr + pivot_row_offsets, mask=col_mask, other=0.0)
        selected_row = tl.load(work_ptr + selected_row_offsets, mask=col_mask, other=0.0)
        tl.store(work_ptr + pivot_row_offsets, selected_row, mask=col_mask)
        tl.store(work_ptr + selected_row_offsets, pivot_row, mask=col_mask)
        tl.debug_barrier()

        pivot_value = tl.load(work_ptr + k * stride_a_row + k * stride_a_col)
        for row_start in range(k + 1, n, BLOCK_ROWS):
            rows = (row_start + row_ids)[:, None]
            offsets = rows * stride_a_row + cols[None, :] * stride_a_col
            update_mask = (rows < n) & (cols[None, :] >= k) & col_mask[None, :]
            block = tl.load(work_ptr + offsets, mask=update_mask, other=0.0)

            multiplier_offsets = (row_start + row_ids) * stride_a_row + k * stride_a_col
            multiplier_mask = (row_start + row_ids) < n
            multipliers = tl.load(
                work_ptr + multiplier_offsets,
                mask=multiplier_mask,
                other=0.0,
            )
            updated = block - multipliers[:, None] * pivot_row[None, :]
            tl.store(work_ptr + offsets, updated, mask=update_mask)
        tl.debug_barrier()

    for i in range(0, n):
        lower_offsets = i * stride_a_row + cols * stride_a_col
        lower = tl.load(
            work_ptr + lower_offsets,
            mask=(cols < i) & col_mask,
            other=0.0,
        )
        prior = tl.load(x_ptr + cols * stride_x, mask=cols < i, other=0.0)
        value = tl.load(b_ptr + i * stride_b) - tl.sum(lower * prior, axis=0)
        tl.store(x_ptr + i * stride_x, value)
        tl.debug_barrier()

    for i in range(0, n):
        row = n - 1 - i
        upper_offsets = row * stride_a_row + cols * stride_a_col
        upper = tl.load(
            work_ptr + upper_offsets,
            mask=(cols > row) & col_mask,
            other=0.0,
        )
        solved = tl.load(x_ptr + cols * stride_x, mask=cols > row, other=0.0)
        value = tl.load(x_ptr + row * stride_x) - tl.sum(upper * solved, axis=0)
        diagonal = tl.load(work_ptr + row * stride_a_row + row * stride_a_col)
        tl.store(x_ptr + row * stride_x, tl.div_rn(value, diagonal))
        tl.debug_barrier()


def build(context):
    def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be square")
        if b.ndim != 1 or b.shape[0] != A.shape[0]:
            raise ValueError("b must be a vector matching A")

        n = A.shape[0]
        block_n = triton.next_power_of_2(n)
        if block_n > 1024:
            raise ValueError("the fused kernel supports matrices up to size 1024")

        work = torch.empty_like(A)
        x = torch.empty_like(b)
        _fused_lu_solve_kernel[(1,)](
            A,
            b,
            work,
            x,
            n,
            A.stride(0),
            A.stride(1),
            b.stride(0),
            x.stride(0),
            BLOCK_N=block_n,
            BLOCK_ROWS=16,
            num_warps=8,
        )
        return x

    return fused_lu_solve
