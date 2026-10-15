import torch
import intent
import intent.language as I


@intent.kernel
def _solve_kernel(
    A: I.In[I.f32, (256, 256)],
    B: I.In[I.f32, (256, 1)],
    X: I.Out[I.f32, (256, 1)],
):
    # The augmented matrix is a kernel-local logical workspace.  The outer
    # elimination loop is ordered; rows within each update are independent.
    aug = I.buffer((256, 257), I.f32)
    for i in range(256):
        for j in range(256):
            aug[i, j] = A[i, j]
        aug[i, 256] = B[i, 0]

    for k in range(256):
        pivot = aug[k, k]
        for j in range(257):
            aug[k, j] = I.fdiv(aug[k, j], pivot)

        for i in I.parallel(I.domain(0, 256)):
            if i != k:
                factor = aug[i, k]
                for j in range(257):
                    aug[i, j] = aug[i, j] - factor * aug[k, j]

    for i in I.parallel(I.domain(0, 256)):
        X[i, 0] = aug[i, 256]


def build(context):
    artifact = context.compile("solve_gauss_jordan", _solve_kernel)

    def solve(A, B, *, left=True, out=None):
        result = out if out is not None else torch.empty_like(B)
        artifact(A, B, result)
        return result

    return solve
