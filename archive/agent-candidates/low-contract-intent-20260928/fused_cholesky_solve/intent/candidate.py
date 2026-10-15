import torch
import intent
import intent.language as I


@intent.kernel
def _cholesky(
    A: I.In[I.f32, ("N", "N")],
    L: I.InOut[I.f32, ("N", "N")],
):
    n = A.shape[0]
    for i in range(n):
        for j in range(i + 1):
            s = I.cast(0.0, I.f32)
            for p in range(j):
                s = s + L[i, p] * L[j, p]
            if i == j:
                L[i, j] = I.sqrt(A[i, i] - s)
            else:
                L[i, j] = (A[i, j] - s) / L[j, j]


@intent.kernel
def _forward(
    L: I.In[I.f32, ("N", "N")],
    b: I.In[I.f32, ("N", 1)],
    y: I.InOut[I.f32, ("N", 1)],
):
    n = L.shape[0]
    for i in range(n):
        s = I.cast(0.0, I.f32)
        for j in range(i):
            s = s + L[i, j] * y[j, 0]
        y[i, 0] = (b[i, 0] - s) / L[i, i]


@intent.kernel
def _backward(
    L: I.In[I.f32, ("N", "N")],
    y: I.In[I.f32, ("N", 1)],
    x: I.InOut[I.f32, ("N", 1)],
):
    n = L.shape[0]
    for offset in range(n):
        i = n - 1 - offset
        s = I.cast(0.0, I.f32)
        for j in range(i + 1, n):
            s = s + L[j, i] * x[j, 0]
        x[i, 0] = (y[i, 0] - s) / L[i, i]


def build(context):
    cholesky = context.compile("fused_cholesky_solve_cholesky", _cholesky)
    forward = context.compile("fused_cholesky_solve_forward", _forward)
    backward = context.compile("fused_cholesky_solve_backward", _backward)

    def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        n = A.shape[0]
        L = torch.empty((n, n), device=A.device, dtype=torch.float32)
        y = torch.empty((n, 1), device=b.device, dtype=torch.float32)
        x = torch.empty((n, 1), device=b.device, dtype=torch.float32)
        cholesky(A, L)
        forward(L, b, y)
        backward(L, y, x)
        return x

    return fused_cholesky_solve
