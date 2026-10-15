import torch
import triton
import triton.language as tl


@triton.jit
def _qr_solve(A, b, out, factor, reflectors, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    # One logical program owns the factorization so the reflector sequence is ordered.
    pid = tl.program_id(0)
    rows = tl.arange(0, M)
    cols = tl.arange(0, N)
    valid_rows = rows < M
    valid_cols = cols < N
    if pid == 0:
        for j in range(N):
            tl.store(factor + rows * N + j, tl.load(A + rows * N + j, mask=valid_rows, other=0.0), mask=valid_rows)
        for k in range(N):
            rmask = (rows >= k) & valid_rows
            x = tl.load(factor + rows * N + k, mask=rmask, other=0.0)
            norm = tl.sqrt(tl.sum(x * x, axis=0))
            x0 = tl.sum(tl.where(rows == k, x, 0.0), axis=0)
            alpha = tl.where(x0 >= 0.0, -norm, norm)
            v = tl.where(rmask, x, 0.0)
            v = tl.where(rows == k, x0 - alpha, v)
            vv = tl.sum(v * v, axis=0)
            tau = 2.0 / vv
            tl.store(reflectors + rows * N + k, v, mask=valid_rows)
            tl.store(factor + rows * N + k, tl.where(rows == k, alpha, 0.0), mask=rmask)
            for j in range(k + 1, N):
                cmask = rows >= k
                z = tl.load(factor + rows * N + j, mask=cmask, other=0.0)
                proj = tau * tl.sum(v * z, axis=0)
                z = tl.where(cmask, z - v * proj, z)
                tl.store(factor + rows * N + j, z, mask=cmask)

        for rhs in range(K):
            y = tl.load(b + rows * K + rhs, mask=valid_rows, other=0.0)
            for k in range(N):
                v = tl.load(reflectors + rows * N + k, mask=valid_rows, other=0.0)
                vv = tl.sum(v * v, axis=0)
                proj = (2.0 / vv) * tl.sum(v * y, axis=0)
                y = y - v * proj
            for kk in range(N):
                k = N - 1 - kk
                if k < N:
                    diag = tl.load(factor + k * N + k)
                    s = tl.sum(tl.where(rows > k, tl.load(factor + rows * N + k, mask=rows < M, other=0.0) * y, 0.0), axis=0)
                    yk = (tl.sum(tl.where(rows == k, y, 0.0), axis=0) - s) / diag
                    y = tl.where(rows == k, yk, y)
            for i in range(N):
                value = tl.sum(tl.where(rows == i, y, 0.0), axis=0)
                tl.store(out + i * K + rhs, value)


def build(context):
    def fused_qr_solve(A, b):
        m, n = A.shape
        k = b.shape[1]
        work = torch.empty((m, n), device=A.device, dtype=A.dtype)
        reflectors = torch.empty((m, n), device=A.device, dtype=A.dtype)
        out = torch.empty((n, k), device=A.device, dtype=A.dtype)
        _qr_solve[(1,)](A, b, out, work, reflectors, M=m, N=n, K=k)
        return out
    return fused_qr_solve
