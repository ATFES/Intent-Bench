import torch
import triton
import triton.language as tl


@triton.jit
def _det_qr_kernel(a, out, N: tl.constexpr):
    """One-program Householder QR for the fixed small square matrix."""
    i = tl.arange(0, N)
    j = tl.arange(0, N)
    rows = i[:, None]
    cols = j[None, :]
    r = tl.load(a + rows * N + cols)

    qdet = 1.0
    for k in range(N):
        active = i >= k
        x = tl.where(active, r[:, k], 0.0)
        norm = tl.sqrt(tl.sum(x * x, axis=0))
        x0 = r[k, k]
        alpha = tl.where(x0 >= 0.0, -norm, norm)
        v = tl.where(active, x, 0.0)
        v = tl.where(i == k, v - alpha, v)
        vv = tl.sum(v * v, axis=0)
        beta = tl.where(vv > 0.0, 2.0 / vv, 0.0)
        projection = tl.sum(v[:, None] * r, axis=0)
        update = beta * v[:, None] * projection[None, :]
        r = tl.where((rows >= k) & (cols >= k), r - update, r)
        qdet = qdet * -1.0

    determinant = qdet
    for k in range(N):
        determinant = determinant * r[k, k]
    tl.store(out, determinant)


def build(context):
    def wrapper(A, *, mode="reduced", out=None):
        if out is None:
            out = torch.empty((), device=A.device, dtype=A.dtype)
        _det_qr_kernel[(1,)](A, out, N=32)
        return out

    return wrapper
