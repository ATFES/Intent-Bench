import torch
import triton
import triton.language as tl


@triton.jit
def _least_squares_qr_kernel(
    a_ptr,
    b_ptr,
    x_ptr,
    batch,
    a_batch_stride,
    b_batch_stride,
    a_row_stride,
    b_row_stride,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_id = tl.program_id(0)
    if batch_id >= batch:
        return

    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_valid = rows < M
    col_valid = cols < N
    a_offsets = batch_id * a_batch_stride + rows[:, None] * a_row_stride + cols[None, :]
    valid_2d = row_valid[:, None] & col_valid[None, :]
    matrix = tl.load(a_ptr + a_offsets, mask=valid_2d, other=0.0)
    rhs = tl.load(b_ptr + batch_id * b_batch_stride + rows * b_row_stride, mask=row_valid, other=0.0)

    diagonal = tl.zeros([BLOCK_N], dtype=tl.float32)
    beta = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k in range(N):
        active_column = cols == k
        column = tl.sum(tl.where(col_valid[None, :] & active_column[None, :], matrix, 0.0), axis=1)
        pivot = tl.sum(tl.where(row_valid & (rows == k), column, 0.0))
        below = tl.where(rows >= k, column * column, 0.0)
        norm = tl.sqrt(tl.sum(below))
        alpha = tl.where(pivot >= 0.0, -norm, norm)
        reflector = column
        reflector = tl.where(row_valid & (rows == k), column - alpha, reflector)
        reflector_norm = tl.sum(tl.where(row_valid & (rows >= k), reflector * reflector, 0.0))
        coefficient = tl.where(reflector_norm > 0.0, 2.0 / reflector_norm, 0.0)
        diagonal = tl.where(col_valid & active_column, alpha, diagonal)

        projection = tl.sum(tl.where(valid_2d & (rows[:, None] >= k), reflector[:, None] * matrix, 0.0), axis=0)
        update = coefficient * reflector[:, None] * projection[None, :]
        matrix = tl.where(valid_2d & (cols[None, :] >= k), matrix - update, matrix)

        rhs_projection = tl.sum(tl.where(row_valid & (rows >= k), reflector * rhs, 0.0))
        rhs = tl.where(row_valid & (rows >= k), rhs - coefficient * reflector * rhs_projection, rhs)
        beta_value = tl.sum(tl.where(row_valid & (rows == k), rhs, 0.0))
        beta = tl.where(col_valid & active_column, beta_value, beta)

    reciprocal = tl.where((diagonal * diagonal) > 0.0, 1.0 / diagonal, 0.0)
    solution = tl.zeros([BLOCK_N], dtype=tl.float32)
    for j in range(BLOCK_N - 1, -1, -1):
        row = tl.sum(
            tl.where(valid_2d & (rows[:, None] == j) & (cols[None, :] > j), matrix, 0.0),
            axis=0,
        )
        partial = tl.sum(tl.where(col_valid, row * solution, 0.0))
        pivot_beta = tl.sum(tl.where(col_valid & (cols == j), beta, 0.0))
        pivot_reciprocal = tl.sum(tl.where(col_valid & (cols == j), reciprocal, 0.0))
        value = pivot_reciprocal * (pivot_beta - partial)
        solution = tl.where(col_valid & (cols == j), value, solution)

    x_offsets = batch_id * N + cols
    tl.store(x_ptr + x_offsets, solution, mask=col_valid)


def _next_power_of_two(value):
    result = 1
    while result < value:
        result *= 2
    return result


def build(context):
    def least_squares_qr(A, b, *, mode="reduced", out=None):
        if mode not in ("reduced", "complete"):
            raise ValueError("mode must be 'reduced' or 'complete'")
        if A.ndim != 2 or b.ndim not in (1, 2):
            raise ValueError("A must be 2-D and b must be 1-D or 2-D")
        m, n = A.shape
        rhs_rows = b.shape[0]
        if rhs_rows != m:
            raise ValueError("A and b have incompatible shapes")
        if A.dtype != torch.float32 or b.dtype != torch.float32:
            raise ValueError("A and b must be float32")

        block_m = _next_power_of_two(m)
        block_n = _next_power_of_two(n)
        output = out if out is not None else torch.empty((n, 1), device=A.device, dtype=A.dtype)
        if output.numel() != n:
            raise ValueError("out has an incompatible number of elements")

        _least_squares_qr_kernel[(1,)](
            A,
            b,
            output,
            1,
            0,
            0,
            A.stride(0),
            b.stride(0),
            M=m,
            N=n,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
        return output

    return least_squares_qr
