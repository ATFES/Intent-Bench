import torch
import intent
import intent.language as I


@intent.kernel
def _cholesky(
    A: I.In[I.f32, (256, 256)],
    L: I.Out[I.f32, (256, 256)],
):
    # Rows are ordered by the factorization recurrence.  The entries below
    # each diagonal are independent once that row's diagonal is available.
    for i in I.domain(0, 256):
        diag_sum = I.cast(0.0, I.f32)
        for j in I.domain(0, i):
            value = L[i, j]
            diag_sum = diag_sum + value * value

        diagonal = I.sqrt(A[i, i] - diag_sum)
        L[i, i] = diagonal

        for r in I.parallel(I.domain(i + 1, 256)):
            entry_sum = I.cast(0.0, I.f32)
            for j in I.domain(0, i):
                entry_sum = entry_sum + L[r, j] * L[i, j]
            L[r, i] = (A[r, i] - entry_sum) / diagonal


@intent.kernel
def _forward_substitution(
    L: I.In[I.f32, (256, 256)],
    b: I.In[I.f32, (256, 1)],
    y: I.Out[I.f32, (256, 1)],
):
    # Columns are independent work items; rows retain the triangular order.
    for col in I.parallel(I.domain(0, 1)):
        for i in I.domain(0, 256):
            value = b[i, col]
            for j in I.domain(0, i):
                value = value - L[i, j] * y[j, col]
            y[i, col] = value / L[i, i]


@intent.kernel
def _backward_substitution(
    L: I.In[I.f32, (256, 256)],
    y: I.In[I.f32, (256, 1)],
    x: I.Out[I.f32, (256, 1)],
):
    # Use a forward ordinal to spell the reverse dependent traversal.
    for col in I.parallel(I.domain(0, 1)):
        for ordinal in I.domain(0, 256):
            i = 255 - ordinal
            value = y[i, col]
            for j in I.domain(i + 1, 256):
                value = value - L[j, i] * x[j, col]
            x[i, col] = value / L[i, i]


def build(context):
    cholesky = context.compile("fused_cholesky_factor", _cholesky, constexprs={})
    forward = context.compile("fused_forward_substitution", _forward_substitution, constexprs={})
    backward = context.compile("fused_backward_substitution", _backward_substitution, constexprs={})

    def fused_cholesky_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        L = torch.empty((256, 256), device=A.device, dtype=torch.float32)
        y = torch.empty((256, 1), device=b.device, dtype=torch.float32)
        x = torch.empty((256, 1), device=b.device, dtype=torch.float32)
        cholesky(A, L)
        forward(L, b, y)
        backward(L, y, x)
        return x

    return fused_cholesky_solve
