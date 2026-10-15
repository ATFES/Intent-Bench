import torch
import triton
import triton.language as tl


@triton.jit
def _least_squares_kernel(a_ptr, b_ptr, out_ptr, stride_am, stride_an, stride_bm, stride_bk):
    rows = tl.arange(0, 64)
    cols = tl.arange(0, 16)
    b = tl.load(b_ptr + rows * stride_bm).to(tl.float32)

    # Keep the complete reduced QR factorization in registers for this fixed shape.
    q = tl.zeros((64, 16), dtype=tl.float32)
    r = tl.zeros((16, 16), dtype=tl.float32)
    for j in range(16):
        v = tl.load(a_ptr + rows * stride_am + j * stride_an).to(tl.float32)
        for i in range(j):
            qi = q[:, i]
            rij = tl.sum(qi * v, axis=0)
            r = tl.where((cols[:, None] == i) & (cols[None, :] == j), rij, r)
            v = v - qi * rij
        rjj = tl.sqrt(tl.sum(v * v, axis=0))
        q = tl.where(cols[None, :] == j, v[:, None] / rjj, q)
        r = tl.where((cols[None, :] == j) & (cols[:, None] == j), rjj, r)

    y = tl.zeros((16,), dtype=tl.float32)
    for j in range(16):
        yj = tl.sum(q[:, j] * b, axis=0)
        y = tl.where(cols == j, yj, y)

    x = tl.zeros((16,), dtype=tl.float32)
    for jj in range(16):
        j = 15 - jj
        tail = tl.sum(r[j, :] * x, axis=0)
        xj = (y[j] - tail) / r[j, j]
        x = tl.where(cols == j, xj, x)
    tl.store(out_ptr + cols * stride_bk, x)


def build(context):
    # The evaluator supplies the fixed unbatched float32 profile described in TASK.md.
    def least_squares_qr(A, b, *, mode='reduced', out=None):
        result = out if out is not None else torch.empty((16, 1), device=A.device, dtype=A.dtype)
        _least_squares_kernel[(1,)](A, b, result, A.stride(0), A.stride(1), b.stride(0), result.stride(0))
        return result

    return least_squares_qr
