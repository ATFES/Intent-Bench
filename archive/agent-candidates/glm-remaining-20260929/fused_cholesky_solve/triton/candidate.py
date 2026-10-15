from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _factor_diagonal_kernel(
    source_ptr,
    factor_ptr,
    block_start,
    block_size,
    n,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, BLOCK)
    col = tl.arange(0, BLOCK)

    for local_j in range(block_size):
        prior_col = tl.arange(0, BLOCK)
        prior_mask = prior_col < local_j
        prior_row = tl.load(
            source_ptr + (block_start + local_j) * n + block_start + prior_col,
            mask=prior_mask,
            other=0.0,
        )
        diagonal_value = tl.load(
            source_ptr + (block_start + local_j) * n + block_start + local_j
        )
        diagonal = tl.sqrt_rn(
            diagonal_value - tl.sum(prior_row * prior_row, axis=0)
        )
        tl.store(
            factor_ptr + (block_start + local_j) * n + block_start + local_j,
            diagonal,
        )

        below_mask = (row > local_j) & (row < block_size)
        prior_panel = tl.load(
            source_ptr
            + (block_start + row[:, None]) * n
            + block_start
            + prior_col[None, :],
            mask=below_mask[:, None] & prior_mask[None, :],
            other=0.0,
        )
        column_value = tl.load(
            source_ptr + (block_start + row) * n + block_start + local_j,
            mask=below_mask,
            other=0.0,
        )
        updated = column_value - tl.sum(
            prior_panel * prior_row[None, :], axis=1
        )
        column = updated / diagonal
        tl.store(
            factor_ptr + (block_start + row) * n + block_start + local_j,
            column,
            mask=below_mask,
        )


@triton.jit
def _solve_panel_kernel(
    factor_ptr,
    block_start,
    block_size,
    n,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.arange(0, ROWS)
    col = tl.arange(0, BLOCK)
    row_mask = row < n - block_start - block_size

    for local_j in range(block_size):
        diagonal = tl.load(
            factor_ptr
            + (block_start + local_j) * n
            + block_start
            + local_j
        )
        factor_row_mask = col < local_j
        factor_row = tl.load(
            factor_ptr
            + (block_start + local_j) * n
            + block_start
            + col,
            mask=factor_row_mask,
            other=0.0,
        )
        prior_x = tl.load(
            factor_ptr
            + (block_start + block_size + row[:, None]) * n
            + block_start
            + col[None, :],
            mask=row_mask[:, None] & factor_row_mask[None, :],
            other=0.0,
        )
        rhs = tl.load(
            factor_ptr
            + (block_start + block_size + row) * n
            + block_start
            + local_j,
            mask=row_mask,
            other=0.0,
        )
        x = (rhs - tl.sum(prior_x * factor_row[None, :], axis=1)) / diagonal
        tl.store(
            factor_ptr
            + (block_start + block_size + row) * n
            + block_start
            + local_j,
            x,
            mask=row_mask,
        )


@triton.jit
def _update_trailing_kernel(
    factor_ptr,
    panel_start,
    panel_end,
    n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_id = tl.program_id(0)
    tiles_n = tl.cdiv(n - panel_end, BLOCK_N)
    tile_m = tile_id // tiles_n
    tile_n = tile_id % tiles_n
    row = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, BLOCK_K)
    row_mask = row < n - panel_end
    col_mask = col < n - panel_end

    panel_rows = panel_end + row[:, None]
    panel_cols = panel_start + inner[None, :]
    left = tl.load(
        factor_ptr + panel_rows * n + panel_cols,
        mask=row_mask[:, None],
        other=0.0,
    )
    transposed_rows = panel_start + inner[:, None]
    transposed_cols = panel_end + col[None, :]
    right = tl.load(
        factor_ptr + transposed_rows * n + transposed_cols,
        mask=col_mask[None, :],
        other=0.0,
    )
    update = tl.dot(
        left,
        right,
        input_precision="ieee",
        out_dtype=tl.float32,
    )
    destination = factor_ptr + panel_rows * n + panel_end + col[None, :]
    old = tl.load(destination, mask=row_mask[:, None] & col_mask[None, :], other=0.0)
    tl.store(
        destination,
        old - update,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _cholesky_solve_kernel(
    factor_ptr,
    rhs_ptr,
    solution_ptr,
    n,
    k,
    rhs_row_stride,
    rhs_col_stride,
    out_row_stride,
    out_col_stride,
    BLOCK: tl.constexpr,
):
    column = tl.program_id(0)
    row = tl.arange(0, BLOCK)

    for i in range(n):
        prior_mask = row < i
        factor_row = tl.load(
            factor_ptr + i * n + row, mask=prior_mask, other=0.0
        )
        prior_y = tl.load(
            solution_ptr + row * out_row_stride + column * out_col_stride,
            mask=prior_mask,
            other=0.0,
        )
        rhs = tl.load(rhs_ptr + i * rhs_row_stride + column * rhs_col_stride)
        diagonal = tl.load(factor_ptr + i * n + i)
        y = (rhs - tl.sum(factor_row * prior_y, axis=0)) / diagonal
        tl.store(
            solution_ptr + i * out_row_stride + column * out_col_stride, y
        )

    for i in tl.range(n - 1, -1, -1):
        later_mask = row > i
        later_mask = later_mask & (row < n)
        factor_col = tl.load(
            factor_ptr + row * n + i, mask=later_mask, other=0.0
        )
        later_x = tl.load(
            solution_ptr + row * out_row_stride + column * out_col_stride,
            mask=later_mask,
            other=0.0,
        )
        current = tl.load(
            solution_ptr + i * out_row_stride + column * out_col_stride
        )
        diagonal = tl.load(factor_ptr + i * n + i)
        x = (current - tl.sum(factor_col * later_x, axis=0)) / diagonal
        tl.store(
            solution_ptr + i * out_row_stride + column * out_col_stride, x
        )


def build(context):
    del context

    def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square two-dimensional tensor")
        if b.ndim != 2 or b.shape[0] != A.shape[0]:
            raise ValueError("b must have shape (A.shape[0], k)")

        n = A.shape[0]
        k = b.shape[1]
        factor = A.new_empty((n, n))
        solution = b.new_empty((n, k))
        block_size = 128
        solve_block = triton.next_power_of_2(max(n, 1))
        panel_rows = triton.next_power_of_2(max(n - block_size, 1))

        for block_start in range(0, n, block_size):
            panel_end = min(block_start + block_size, n)
            current_block_size = panel_end - block_start
            source = A if block_start == 0 else factor
            _factor_diagonal_kernel[(1,)](
                source,
                factor,
                block_start,
                current_block_size,
                n,
                BLOCK=block_size,
            )
            if panel_end < n:
                _solve_panel_kernel[(1,)](
                    factor,
                    block_start,
                    current_block_size,
                    n,
                    ROWS=panel_rows,
                    BLOCK=block_size,
                )
                grid = ((n - panel_end + 63) // 64) ** 2
                _update_trailing_kernel[(grid,)](
                    factor,
                    block_start,
                    panel_end,
                    n,
                    BLOCK_M=64,
                    BLOCK_N=64,
                    BLOCK_K=block_size,
                )

        _cholesky_solve_kernel[(k,)](
            factor,
            b,
            solution,
            n,
            k,
            b.stride(0),
            b.stride(1),
            solution.stride(0),
            solution.stride(1),
            BLOCK=solve_block,
        )
        return solution

    return fused_cholesky_solve
