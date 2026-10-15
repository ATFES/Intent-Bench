from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _copy_matrix_kernel(
    src_ptr, dst_ptr,
    src_row_stride, src_col_stride,
    dst_row_stride, dst_col_stride,
    n_rows, n_cols,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows < n_rows)[:, None] & (cols < n_cols)[None, :]
    values = tl.load(
        src_ptr + rows[:, None] * src_row_stride + cols[None, :] * src_col_stride,
        mask=mask, other=0.0,
    )
    tl.store(
        dst_ptr + rows[:, None] * dst_row_stride + cols[None, :] * dst_col_stride,
        values, mask=mask,
    )


@triton.jit
def _factor_panel_kernel(
    lu_ptr, rhs_ptr,
    lu_row_stride, lu_col_stride,
    rhs_row_stride, rhs_col_stride,
    n, base,
    PANEL: tl.constexpr, ROW_BLOCK: tl.constexpr, SWAP_BLOCK: tl.constexpr,
):
    rows = tl.arange(0, ROW_BLOCK)
    row_index = tl.arange(0, ROW_BLOCK)

    for local_step in tl.static_range(PANEL):
        col = base + local_step
        col_values = tl.load(
            lu_ptr + rows * lu_row_stride + col * lu_col_stride,
            mask=rows >= col, other=float("-inf"),
        )
        pivot_row = tl.argmax(col_values, axis=0, tie_break_left=True)
        panel_columns = base + tl.arange(0, PANEL)
        current_panel = tl.load(
            lu_ptr + col * lu_row_stride + panel_columns * lu_col_stride
        )
        pivot_panel = tl.load(
            lu_ptr + pivot_row * lu_row_stride + panel_columns * lu_col_stride
        )
        tl.store(lu_ptr + col * lu_row_stride + panel_columns * lu_col_stride, pivot_panel)
        tl.store(lu_ptr + pivot_row * lu_row_stride + panel_columns * lu_col_stride, current_panel)

        for swap_base in range(0, n, SWAP_BLOCK):
            swap_cols = swap_base + tl.arange(0, SWAP_BLOCK)
            current_swap = tl.load(lu_ptr + col * lu_row_stride + swap_cols * lu_col_stride)
            pivot_swap = tl.load(lu_ptr + pivot_row * lu_row_stride + swap_cols * lu_col_stride)
            tl.store(lu_ptr + col * lu_row_stride + swap_cols * lu_col_stride, pivot_swap)
            tl.store(lu_ptr + pivot_row * lu_row_stride + swap_cols * lu_col_stride, current_swap)

        column_zero = tl.arange(0, 1)
        current_rhs = tl.load(rhs_ptr + col * rhs_row_stride + column_zero * rhs_col_stride)
        pivot_rhs = tl.load(rhs_ptr + pivot_row * rhs_row_stride + column_zero * rhs_col_stride)
        tl.store(rhs_ptr + col * rhs_row_stride + column_zero * rhs_col_stride, pivot_rhs)
        tl.store(rhs_ptr + pivot_row * rhs_row_stride + column_zero * rhs_col_stride, current_rhs)

        tl.debug_barrier()
        panel = tl.load(
            lu_ptr + row_index[:, None] * lu_row_stride + panel_columns[None, :] * lu_col_stride
        )
        panel_column = tl.sum(tl.where(row_index[:, None] == col, panel, 0.0), axis=0)
        pivot_value = tl.sum(tl.where(row_index == col, panel_column, 0.0), axis=0)
        scale = 1.0 / pivot_value
        lower = row_index > col
        panel_column = tl.where(
            row_index == col, 1.0, tl.where(lower, panel_column * scale, panel_column)
        )
        pivot_panel = tl.sum(tl.where(row_index[:, None] == col, panel, 0.0), axis=0)
        panel = tl.where(
            lower[:, None], panel - panel_column[:, None] * pivot_panel[None, :], panel
        )
        tl.store(
            lu_ptr + row_index[:, None] * lu_row_stride + panel_columns[None, :] * lu_col_stride,
            panel,
        )
        tl.debug_barrier()


@triton.jit
def _update_trailing_kernel(
    lu_ptr, lu_row_stride, lu_col_stride,
    n, start, PANEL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = start + pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = start + pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    factors = start + tl.arange(0, PANEL)
    row_valid = rows < n
    col_valid = cols < n
    mask = row_valid[:, None] & col_valid[None, :]
    lower = tl.load(
        lu_ptr + rows[:, None] * lu_row_stride + factors[None, :] * lu_col_stride,
        mask=row_valid[:, None], other=0.0,
    )
    upper = tl.load(
        lu_ptr + factors[:, None] * lu_row_stride + cols[None, :] * lu_col_stride,
        mask=col_valid[None, :], other=0.0,
    )
    update = tl.dot(lower, upper, input_precision="ieee", out_dtype=tl.float32)
    destination = tl.load(
        lu_ptr + rows[:, None] * lu_row_stride + cols[None, :] * lu_col_stride,
        mask=mask, other=0.0,
    )
    tl.store(
        lu_ptr + rows[:, None] * lu_row_stride + cols[None, :] * lu_col_stride,
        destination - update, mask=mask,
    )


@triton.jit
def _triangular_solve_kernel(
    lu_ptr, rhs_ptr, out_ptr,
    lu_row_stride, lu_col_stride,
    rhs_row_stride, rhs_col_stride,
    out_row_stride, out_col_stride,
    n, ROW_BLOCK: tl.constexpr,
):
    rows = tl.arange(0, ROW_BLOCK)
    cols = tl.arange(0, ROW_BLOCK)
    column_zero = tl.arange(0, 1)
    solution = tl.load(rhs_ptr + rows * rhs_row_stride + column_zero * rhs_col_stride)

    for col in range(0, n - 1):
        current = tl.sum(tl.where(rows == col, solution, 0.0), axis=0)
        tl.store(out_ptr + col * out_row_stride + column_zero * out_col_stride, current)
        multiplier = tl.load(
            lu_ptr + rows * lu_row_stride + col * lu_col_stride,
            mask=rows > col, other=0.0,
        )
        solution = solution - multiplier * current

    last = tl.sum(tl.where(rows == n - 1, solution, 0.0), axis=0)
    tl.store(out_ptr + (n - 1) * out_row_stride + column_zero * out_col_stride, last)

    for reverse_step in range(0, n):
        col = n - 1 - reverse_step
        row_values = tl.load(
            lu_ptr + col * lu_row_stride + cols * lu_col_stride,
            mask=cols > col, other=0.0,
        )
        known = tl.load(out_ptr + rows * out_row_stride + column_zero * out_col_stride)
        known = tl.where(rows > col, known, 0.0)
        rhs = tl.sum(tl.where(rows == col, known, 0.0), axis=0)
        correction = tl.sum(row_values * known, axis=0)
        diagonal = tl.load(lu_ptr + col * lu_row_stride + col * lu_col_stride)
        value = (rhs - correction) / diagonal
        tl.store(out_ptr + col * out_row_stride + column_zero * out_col_stride, value)


def build(context):
    def solve(A: torch.Tensor, B: torch.Tensor, *, left: bool = True, out: torch.Tensor | None = None):
        if not left:
            raise NotImplementedError("only left=True is supported by this profile")
        if A.device.type != "cuda" or B.device.type != "cuda":
            raise ValueError("A and B must be CUDA tensors")
        if A.dtype is not torch.float32 or B.dtype is not torch.float32:
            raise ValueError("only float32 is supported by this profile")
        if (
            A.ndim != 2 or B.ndim != 2
            or A.shape[0] != A.shape[1]
            or B.shape[0] != A.shape[0] or B.shape[1] != 1
        ):
            raise ValueError("profile requires A[n,n] and B[n,1]")

        n = A.shape[0]
        if n > 256:
            raise ValueError("profile requires n <= 256")
        work = torch.empty((n, n), dtype=A.dtype, device=A.device)
        pivoted_rhs = torch.empty((n, 1), dtype=B.dtype, device=B.device)
        result = out if out is not None else torch.empty_like(B)
        _copy_matrix_kernel[(1, 1)](
            B, pivoted_rhs,
            B.stride(0), B.stride(1),
            pivoted_rhs.stride(0), pivoted_rhs.stride(1),
            n, 1,
            BLOCK_M=256, BLOCK_N=1, num_warps=4,
        )
        _copy_matrix_kernel[(triton.cdiv(n, 64), triton.cdiv(n, 64))](
            A, work,
            A.stride(0), A.stride(1),
            work.stride(0), work.stride(1),
            n, n,
            BLOCK_M=64, BLOCK_N=64, num_warps=4,
        )

        panel_width = 32
        for panel_base in range(0, n, panel_width):
            _factor_panel_kernel[(1,)](
                work, pivoted_rhs,
                work.stride(0), work.stride(1),
                pivoted_rhs.stride(0), pivoted_rhs.stride(1),
                n, panel_base,
                PANEL=panel_width, ROW_BLOCK=256, SWAP_BLOCK=64, num_warps=8,
            )
            trailing_start = panel_base + panel_width
            if trailing_start < n:
                remaining = n - trailing_start
                _update_trailing_kernel[(triton.cdiv(remaining, 32), triton.cdiv(remaining, 32))](
                    work, work.stride(0), work.stride(1),
                    n, trailing_start,
                    PANEL=panel_width,
                    BLOCK_M=32, BLOCK_N=32, num_warps=4,
                )

        _triangular_solve_kernel[(1,)](
            work, pivoted_rhs, result,
            work.stride(0), work.stride(1),
            pivoted_rhs.stride(0), pivoted_rhs.stride(1),
            result.stride(0), result.stride(1),
            n, ROW_BLOCK=256, num_warps=8,
        )
        return result

    return solve
