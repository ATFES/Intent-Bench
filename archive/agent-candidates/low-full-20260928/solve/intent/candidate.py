import torch
import intent
import intent.language as I


@intent.kernel
def solve_kernel(
    A: I.In[I.f32, ("N", "N")],
    B: I.In[I.f32, ("N", "R")],
    X: I.Out[I.f32, ("N", "R")],
):
    N, _ = A.shape
    _, R = B.shape
    # The augmented matrix is kept in kernel-local storage.  Pivot steps are
    # ordered; all rows below a pivot are independent once that pivot row is
    # complete.
    mat = I.buffer(("N", "N"), I.f32, init=0.0)
    rhs = I.buffer(("N", "R"), I.f32, init=0.0)

    for i in I.parallel(I.domain(0, N)):
        for j in I.parallel(I.domain(0, N)):
            mat[i, j] = A[i, j]
        for q in I.parallel(I.domain(0, R)):
            rhs[i, q] = B[i, q]

    for k in range(N):
        pivot = mat[k, k]
        for j in I.parallel(I.domain(k + 1, N)):
            mat[k, j] = mat[k, j] / pivot
        mat[k, k] = I.cast(1.0, I.f32)
        for q in I.parallel(I.domain(0, R)):
            rhs[k, q] = rhs[k, q] / pivot

        for i in I.parallel(I.domain(k + 1, N)):
            factor = mat[i, k]
            for j in I.parallel(I.domain(k + 1, N)):
                mat[i, j] = mat[i, j] - factor * mat[k, j]
            mat[i, k] = I.cast(0.0, I.f32)
            for q in I.parallel(I.domain(0, R)):
                rhs[i, q] = rhs[i, q] - factor * rhs[k, q]

    for kk in range(N):
        k = N - 1 - kk
        for q in I.parallel(I.domain(0, R)):
            value = rhs[k, q]
            for j in I.parallel(I.domain(k + 1, N)):
                value = value - mat[k, j] * rhs[j, q]
            rhs[k, q] = value

    for i in I.parallel(I.domain(0, N)):
        for q in I.parallel(I.domain(0, R)):
            X[i, q] = rhs[i, q]


def build(context):
    artifact = context.compile("solve_gaussian", solve_kernel)

    def solve(A, B, *, left=True, out=None):
        result = out if out is not None else torch.empty_like(B)
        artifact(A, B, result)
        return result

    return solve
