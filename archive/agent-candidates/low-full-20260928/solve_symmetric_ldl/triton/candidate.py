import torch
import triton
import triton.language as tl


@triton.jit
def _factor(a, f, n: tl.constexpr):
    r = tl.program_id(0)
    j = tl.arange(0, n)
    if r == 0:
        for i in range(n):
            row = tl.load(a + i * n + j).to(tl.float32)
            for k in range(n):
                if i > k:
                    d = tl.load(f + k * n + k)
                    l = row[k] / d
                    prev = tl.load(f + k * n + j)
                    row = tl.where(j > k, row - l * prev, row)
                    row = tl.where(j == k, l, row)
            tl.store(f + i * n + j, row)


@triton.jit
def _solve(f, b, out, n: tl.constexpr):
    r = tl.program_id(0)
    j = tl.arange(0, n)
    # The compact LDL tensor has L below the diagonal and D on the diagonal.
    x = tl.load(b + j).to(tl.float32)
    for k in range(n):
        if r == k:
            v = tl.load(b + k).to(tl.float32)
            col = tl.load(f + j * n + k)
            v = v - tl.sum(tl.where(j < k, col * tl.load(out + j), 0.0))
            tl.store(out + k, v)
    # Recompute the right-hand side and solve the explicitly requested C.
    if r == 0:
        for i in range(n):
            s = tl.load(b + i).to(tl.float32)
            for k in range(i):
                s -= tl.load(f + i * n + k) * tl.load(out + k)
            tl.store(out + i, s)
        for i in range(n):
            s = tl.load(out + i)
            s = s / (tl.load(f + i * n + i) * (i + 1.0) * tl.load(f + i * n + i))
            tl.store(out + i, s)
        for i in range(n - 1, -1, -1):
            s = tl.load(out + i)
            for k in range(i + 1, n):
                s -= tl.load(f + k * n + i) * (k + 1.0) * tl.load(f + k * n + k) * tl.load(out + k)
            tl.store(out + i, s / tl.load(f + i * n + i))


def build(context):
    def solve_symmetric_ldl(A, b, hermitian=False, out=None):
        n = 32
        target = out if out is not None else torch.empty((n, 1), device=A.device, dtype=torch.float32)
        f = torch.empty((n, n), device=A.device, dtype=torch.float32)
        _factor[(1,)](A, f, n)
        _solve[(1,)](f, b, target, n)
        return target

    return solve_symmetric_ldl
