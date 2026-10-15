import torch
import triton
import triton.language as tl


@triton.jit
def _solve_left_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    lu_ptr,
    n,
    a_stride0,
    a_stride1,
    b_stride0,
    out_stride0,
    BLOCK: tl.constexpr,
):
    # There is deliberately one program for this small, fixed-size system.
    # It gives the row operations a strict order without inter-program
    # synchronization, while the row arithmetic itself remains vectorized.
    cols = tl.arange(0, BLOCK)
    rows = cols

    # Copy A to private working storage and seed the RHS.  The copy also
    # makes out=A safe: all reads of A are complete before the solve writes it.
    for i in tl.range(0, n, loop_unroll_factor=1):
        row = tl.load(a_ptr + i * a_stride0 + cols * a_stride1)
        tl.store(lu_ptr + i * n + cols, row)
    rhs = tl.load(b_ptr + rows * b_stride0)
    tl.store(out_ptr + rows * out_stride0, rhs)

    # In-place LU with partial pivoting.  The RHS is permuted alongside A.
    for k in tl.range(0, n, loop_unroll_factor=1):
        column = tl.load(lu_ptr + rows * n + k)
        magnitude = tl.where(rows >= k, tl.abs(column), -float("inf"))
        pivot = tl.argmax(magnitude, axis=0)

        pivot_row = tl.load(lu_ptr + pivot * n + cols)
        current_row = tl.load(lu_ptr + k * n + cols)
        tl.store(lu_ptr + k * n + cols, pivot_row)
        tl.store(lu_ptr + pivot * n + cols, current_row)

        pivot_rhs = tl.load(out_ptr + pivot * out_stride0)
        current_rhs = tl.load(out_ptr + k * out_stride0)
        tl.store(out_ptr + k * out_stride0, pivot_rhs)
        tl.store(out_ptr + pivot * out_stride0, current_rhs)

        pivot_value = tl.load(lu_ptr + k * n + k)
        for i in tl.range(0, n, loop_unroll_factor=1):
            active = i > k
            factor = tl.load(lu_ptr + i * n + k) / pivot_value
            row_i = tl.load(lu_ptr + i * n + cols)
            updated = row_i - factor * pivot_row
            row_i = tl.where(active & (cols > k), updated, row_i)
            row_i = tl.where(active & (cols == k), factor, row_i)
            tl.store(lu_ptr + i * n + cols, row_i)

    # Forward substitution for L y = P B.  out holds y while it is built.
    for i in tl.range(0, n, loop_unroll_factor=1):
        row_i = tl.load(lu_ptr + i * n + cols)
        y = tl.load(out_ptr + cols * out_stride0)
        below_diagonal = cols < i
        accum = tl.sum(tl.where(below_diagonal, row_i * y, 0.0), axis=0)
        value = tl.load(out_ptr + i * out_stride0) - accum
        tl.store(out_ptr + i * out_stride0, value)

    # Back substitution for U X = y.
    for reverse_i in tl.range(0, n, loop_unroll_factor=1):
        i = n - 1 - reverse_i
        row_i = tl.load(lu_ptr + i * n + cols)
        x = tl.load(out_ptr + cols * out_stride0)
        above_diagonal = cols > i
        accum = tl.sum(tl.where(above_diagonal, row_i * x, 0.0), axis=0)
        diagonal = tl.load(lu_ptr + i * n + i)
        value = (tl.load(out_ptr + i * out_stride0) - accum) / diagonal
        tl.store(out_ptr + i * out_stride0, value)


@triton.jit
def _solve_right_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    lu_ptr,
    n,
    a_stride0,
    a_stride1,
    b_stride0,
    b_stride1,
    out_stride1,
    BLOCK: tl.constexpr,
):
    # X A = B is equivalent to A^T X^T = B^T.  This is the same algorithm,
    # with the logical transpose represented by strided loads and stores.
    cols = tl.arange(0, BLOCK)
    rows = cols

    for i in tl.range(0, n, loop_unroll_factor=1):
        row = tl.load(a_ptr + cols * a_stride0 + i * a_stride1)
        tl.store(lu_ptr + i * n + cols, row)
    rhs = tl.load(b_ptr + rows * b_stride1 + 0 * b_stride0)
    tl.store(out_ptr + rows * out_stride1, rhs)

    for k in tl.range(0, n, loop_unroll_factor=1):
        column = tl.load(lu_ptr + rows * n + k)
        magnitude = tl.where(rows >= k, tl.abs(column), -float("inf"))
        pivot = tl.argmax(magnitude, axis=0)

        pivot_row = tl.load(lu_ptr + pivot * n + cols)
        current_row = tl.load(lu_ptr + k * n + cols)
        tl.store(lu_ptr + k * n + cols, pivot_row)
        tl.store(lu_ptr + pivot * n + cols, current_row)

        pivot_rhs = tl.load(out_ptr + pivot * out_stride1)
        current_rhs = tl.load(out_ptr + k * out_stride1)
        tl.store(out_ptr + k * out_stride1, pivot_rhs)
        tl.store(out_ptr + pivot * out_stride1, current_rhs)

        pivot_value = tl.load(lu_ptr + k * n + k)
        for i in tl.range(0, n, loop_unroll_factor=1):
            active = i > k
            factor = tl.load(lu_ptr + i * n + k) / pivot_value
            row_i = tl.load(lu_ptr + i * n + cols)
            updated = row_i - factor * pivot_row
            row_i = tl.where(active & (cols > k), updated, row_i)
            row_i = tl.where(active & (cols == k), factor, row_i)
            tl.store(lu_ptr + i * n + cols, row_i)

    for i in tl.range(0, n, loop_unroll_factor=1):
        row_i = tl.load(lu_ptr + i * n + cols)
        y = tl.load(out_ptr + cols * out_stride1)
        below_diagonal = cols < i
        accum = tl.sum(tl.where(below_diagonal, row_i * y, 0.0), axis=0)
        value = tl.load(out_ptr + i * out_stride1) - accum
        tl.store(out_ptr + i * out_stride1, value)

    for reverse_i in tl.range(0, n, loop_unroll_factor=1):
        i = n - 1 - reverse_i
        row_i = tl.load(lu_ptr + i * n + cols)
        x = tl.load(out_ptr + cols * out_stride1)
        above_diagonal = cols > i
        accum = tl.sum(tl.where(above_diagonal, row_i * x, 0.0), axis=0)
        diagonal = tl.load(lu_ptr + i * n + i)
        value = (tl.load(out_ptr + i * out_stride1) - accum) / diagonal
        tl.store(out_ptr + i * out_stride1, value)


def build(context):
    def wrapper(A, B, *, left=True, out=None):
        if out is None:
            out = torch.empty_like(B)
        lu = torch.empty_like(A)
        n = A.shape[-1]

        if left:
            _solve_left_kernel[(1,)](
                A,
                B,
                out,
                lu,
                n,
                A.stride(0),
                A.stride(1),
                B.stride(0),
                out.stride(0),
                BLOCK=256,
                num_warps=8,
            )
        else:
            _solve_right_kernel[(1,)](
                A,
                B,
                out,
                lu,
                n,
                A.stride(0),
                A.stride(1),
                B.stride(0),
                B.stride(1),
                out.stride(1),
                BLOCK=256,
                num_warps=8,
            )
        return out

    return wrapper
