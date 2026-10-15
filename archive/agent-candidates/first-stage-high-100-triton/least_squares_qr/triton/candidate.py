from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _least_squares_qr_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    a_stride0,
    a_stride1,
    b_stride0,
    b_stride1,
    out_stride0,
    out_stride1,
    M: tl.constexpr,
    N: tl.constexpr,
):
    # The whole fixed-size problem is handled by one program.  Keeping the
    # factorization in registers avoids materializing Q or any intermediate
    # products in global memory.
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    a_offsets = rows[:, None] * a_stride0 + cols[None, :] * a_stride1
    a = tl.load(a_ptr + a_offsets)
    b = tl.load(b_ptr + rows * b_stride0)

    # Householder QR.  The strictly lower part of a is temporary reflector
    # storage; after each step the active trailing matrix is updated in place.
    for j in range(N):
        row_active = rows >= j
        col_j = tl.sum(a * (cols[None, :] == j), axis=1)
        v = tl.where(row_active, col_j, 0.0)
        x0 = tl.sum(v * (rows == j), axis=0)
        norm = tl.sqrt(tl.sum(v * v, axis=0))
        alpha = tl.where(x0 >= 0.0, -norm, norm)
        v = v + tl.where(rows == j, alpha, 0.0)
        vv = tl.sum(v * v, axis=0)
        tau = 2.0 / vv

        active = row_active[:, None] & (cols[None, :] >= j)
        projection = tau * tl.sum(v[:, None] * a, axis=0)
        a = tl.where(active, a - v[:, None] * projection[None, :], a)

        b_projection = tau * tl.sum(v * b, axis=0)
        b = b - v * b_projection

    # Solve R x = Q^T b, using the upper triangle left in a.
    x = tl.zeros((N,), dtype=tl.float32)
    for i in range(N - 1, -1, -1):
        row_i = tl.sum(a * (rows[:, None] == i), axis=0)
        tail = tl.sum(row_i * x * (cols > i), axis=0)
        diagonal = tl.sum(row_i * (cols == i), axis=0)
        b_i = tl.sum(b * (rows == i), axis=0)
        xi = (b_i - tail) / diagonal
        x = tl.where(cols == i, xi, x)

    out_offsets = cols * out_stride0
    tl.store(out_ptr + out_offsets, x)


def build(context):
    def wrapper(A, b, *, mode="reduced", out=None):
        if out is None:
            out = torch.empty((16, 1), device=A.device, dtype=A.dtype)
        _least_squares_qr_kernel[(1,)](
            A,
            b,
            out,
            A.stride(0),
            A.stride(1),
            b.stride(0),
            b.stride(1),
            out.stride(0),
            out.stride(1),
            M=64,
            N=16,
            num_warps=4,
        )
        return out

    return wrapper
