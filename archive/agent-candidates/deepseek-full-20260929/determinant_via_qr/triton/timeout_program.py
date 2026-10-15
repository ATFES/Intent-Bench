from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _mul_reduce(x, y):
    return x * y


@triton.jit
def _qr_det_kernel(A, out, N: tl.constexpr):
    # One program owns the whole N x N matrix.  The fixed profile is N = 32.
    rows = tl.expand_dims(tl.arange(0, N), 1)
    cols = tl.expand_dims(tl.arange(0, N), 0)

    # A is row-major and contiguous for the fixed invocation.
    a = tl.load(A + rows * N + cols)

    # Householder QR.  For k = 0 .. N-1:
    #   alpha = -sign(x_k) ||x||
    #   a[k:, k:] <- (I - tau * v v^T) a[k:, k:]
    # where v is the LAPACK-style normalized vector with v[k] == 1.
    for k in tl.static_range(N):
        col_mask = cols == k
        x = tl.sum(tl.where(col_mask, a, 0.0), axis=1, keep_dims=True)

        low = rows >= k
        xm = tl.where(low, x, 0.0)
        xk = tl.sum(tl.where(rows == k, x, 0.0))

        # Stable Euclidean norm of the active sub-column.
        x_abs = tl.abs(xm)
        x_max = tl.max(x_abs)
        scale = tl.where(x_max > 0.0, x_max, 1.0)
        scaled = xm / scale
        norm = scale * tl.sqrt_rn(tl.sum(scaled * scaled))

        alpha = -tl.where(xk >= 0.0, norm, -norm)
        u0 = xk - alpha
        u0_safe = tl.where(norm > 0.0, u0, 1.0)

        # LAPACK-style normalized Householder vector (v[k] == 1).
        v = tl.where(rows == k, 1.0, xm / u0_safe)
        tau = tl.where(norm > 0.0, 1.0 + tl.abs(xk) / norm, 0.0)

        s = tl.sum(v * a, axis=0, keep_dims=True)
        update = tau * (v * s)

        valid = (rows >= k) & (cols >= k)
        a = tl.where(valid, a - update, a)

    # det(A) = det(Q) * prod(diag(R)).  With Householder reflectors,
    # det(Q) = (-1)^(number of reflectors) = (-1)^N for any invertible A.
    # If A is singular, prod(diag(R)) is already zero, so the sign is irrelevant.
    diag = tl.sum(tl.where(rows == cols, a, 0.0), axis=1)
    prod = tl.reduce(diag, 0, _mul_reduce)

    if N % 2 == 1:
        prod = -prod

    tl.store(out, prod)


def build(context):
    def determinant_via_qr(A, *, mode='reduced', out=None):
        n = A.shape[0]
        result = out if out is not None else torch.empty((), device=A.device, dtype=A.dtype)
        _qr_det_kernel[(1,)](A, result, N=n, num_warps=4)
        return result

    return determinant_via_qr
