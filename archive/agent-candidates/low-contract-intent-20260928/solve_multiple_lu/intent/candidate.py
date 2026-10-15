import torch
import intent
import intent.language as I


@intent.kernel
def _solve(A: I.In[I.f32, ("N", "N")], B: I.In[I.f32, ("N", "K")],
           LU: I.Out[I.f32, ("N", "N")], X: I.Out[I.f32, ("N", "K")],
           PIV: I.Out[I.i64, ("N",)], PIVOT: I.Constexpr[bool]):
    n = A.shape[0]
    k = B.shape[1]
    for i in range(n):
        for j in range(n):
            LU[i, j] = A[i, j]
    for i in range(n):
        p = i
        if PIVOT:
            best = I.abs(LU[i, i])
            for r in range(i + 1, n):
                v = I.abs(LU[r, i])
                if v > best:
                    best = v
                    p = r
            PIV[i] = p
            if p != i:
                for j in range(n):
                    t = LU[i, j]
                    LU[i, j] = LU[p, j]
                    LU[p, j] = t
        else:
            PIV[i] = i
        for r in range(i + 1, n):
            LU[r, i] = LU[r, i] / LU[i, i]
            for j in range(i + 1, n):
                LU[r, j] = LU[r, j] - LU[r, i] * LU[i, j]
    for col in I.parallel(I.domain(0, k)):
        for i in range(n):
            v = B[i, col]
            if PIVOT:
                # permutation is applied by swapping the RHS in decomposition order
                for q in range(i):
                    if PIV[q] == i:
                        v = B[q, col]
            for j in range(i):
                v = v - LU[i, j] * X[j, col]
            X[i, col] = v
        for ii in range(n):
            i = n - 1 - ii
            v = X[i, col]
            for j in range(i + 1, n):
                v = v - LU[i, j] * X[j, col]
            X[i, col] = v / LU[i, i]


def build(context):
    kp = context.compile("solve_lu_pivot", _solve, constexprs={"PIVOT": True})
    kn = context.compile("solve_lu_nopivot", _solve, constexprs={"PIVOT": False})

    def solve_multiple_lu(A, Bs, *, pivot=True, out=None):
        if out is None:
            out = torch.empty_like(Bs)
        lu = torch.empty_like(A)
        piv = torch.empty((A.shape[-1],), device=A.device, dtype=torch.int64)
        (kp if pivot else kn)(A, Bs, lu, out, piv)
        return out
    return solve_multiple_lu
