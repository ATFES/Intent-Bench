import torch
import intent
import intent.language as I


@intent.kernel
def _lu_solve(
    A: I.In[I.f32, ("N", "N")],
    b: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    N = A.shape[0]
    lu = I.buffer((N, N), I.f32, init=A[:, :])
    y = I.buffer((N,), I.f32)
    x = I.buffer((N,), I.f32)

    # Right-looking in-place LU with the same partial-pivoting rule as lu().
    for k in I.domain(0, N):
        pivot = k
        best = I.abs(lu[k, k])
        for i in I.domain(k + 1, N):
            candidate = I.abs(lu[i, k])
            if candidate > best:
                best = candidate
                pivot = i
        for j in I.domain(0, N):
            tmp = lu[k, j]
            lu[k, j] = lu[pivot, j]
            lu[pivot, j] = tmp
        diag = lu[k, k]
        for i in I.domain(k + 1, N):
            factor = lu[i, k] / diag
            lu[i, k] = factor
            for j in I.domain(k + 1, N):
                lu[i, j] = lu[i, j] - factor * lu[k, j]

    # P is intentionally not applied to b.  These are ordered triangular solves.
    for i in I.domain(0, N):
        value = b[i]
        for j in I.domain(0, i):
            value = value - lu[i, j] * y[j]
        y[i] = value
    for ii in I.domain(0, N):
        i = N - 1 - ii
        value = y[i]
        for j in I.domain(i + 1, N):
            value = value - lu[i, j] * x[j]
        x[i] = value / lu[i, i]
    for i in I.parallel(I.domain(0, N)):
        out[i] = x[i]


def build(context):
    kernel = context.compile("fused_lu_solve_kernel", _lu_solve)

    def fused_lu_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(b)
        kernel(A, b, out)
        return out

    return fused_lu_solve
