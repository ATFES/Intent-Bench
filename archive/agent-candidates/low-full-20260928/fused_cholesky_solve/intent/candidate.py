import torch
import intent
import intent.language as I


@intent.kernel
def _cholesky(
    A: I.In[I.f32, ("N", "N")],
    L: I.Out[I.f32, ("N", "N")],
):
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            if j < i:
                s = I.cast(0.0, I.f32)
                for q in range(j):
                    s = s + L[i, q] * L[j, q]
                L[i, j] = (A[i, j] - s) / L[j, j]
            elif j == i:
                s = I.cast(0.0, I.f32)
                for q in range(i):
                    s = s + L[i, q] * L[i, q]
                L[i, i] = I.sqrt(A[i, i] - s)


@intent.kernel
def _forward(
    L: I.In[I.f32, ("N", "N")],
    b: I.In[I.f32, ("N", 1)],
    y: I.Out[I.f32, ("N", 1)],
):
    N = L.shape[0]
    for i in range(N):
        s = I.cast(0.0, I.f32)
        for q in range(i):
            s = s + L[i, q] * y[q, 0]
        y[i, 0] = (b[i, 0] - s) / L[i, i]


@intent.kernel
def _backward(
    L: I.In[I.f32, ("N", "N")],
    y: I.In[I.f32, ("N", 1)],
    x: I.Out[I.f32, ("N", 1)],
):
    N = L.shape[0]
    for oi in range(N):
        i = N - 1 - oi
        s = I.cast(0.0, I.f32)
        for q in range(i + 1, N):
            s = s + L[q, i] * x[q, 0]
        x[i, 0] = (y[i, 0] - s) / L[i, i]


def build(context):
    cholesky = context.compile("fused_cholesky_cholesky", _cholesky)
    forward = context.compile("fused_cholesky_forward", _forward)
    backward = context.compile("fused_cholesky_backward", _backward)

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
