import torch
import intent
import intent.language as I


@intent.kernel
def qr_solve_kernel(
    A: I.In[I.f32, (512, 256)],
    b: I.In[I.f32, (512, 10)],
    x: I.Out[I.f32, (256, 10)],
):
    M = 512
    N = 256
    K = 10
    rows = I.domain(0, M)
    cols = I.domain(0, N)
    rhs = I.domain(0, K)

    # work stores both the evolving upper triangle and the lower reflector
    # tails. The two diagonal arrays hold R's diagonal and v[j]'s first entry.
    work = I.buffer((M, N), I.f32, init=A)
    diagonal = I.buffer((N,), I.f32)
    reflector_diagonal = I.buffer((N,), I.f32)
    y = I.buffer((M, K), I.f32, init=b)

    # Householder QR. The column loop is ordered; trailing columns and
    # reductions within each reflector are independent.
    for j in I.domain(0, N):
        tail = rows[j:M]
        column = work[tail, j]
        norm_sq = I.reduce.sum(column * column, axis=0, acc_dtype=I.f32)
        norm = I.sqrt(norm_sq)
        x0 = work[j, j]
        zero = I.cast(0.0, I.f32)
        alpha = I.select(x0 >= zero, -norm, norm)
        coordinates = I.indices(tail)
        vector = I.select(coordinates == j, column - alpha, column)

        vector_sq = I.reduce.sum(vector * vector, axis=0, acc_dtype=I.f32)
        tau = I.cast(0.0, I.f32)
        if norm > zero:
            tau = I.fdiv(I.cast(2.0, I.f32), vector_sq)

        reflector_diagonal[j] = x0 - alpha
        diagonal[j] = alpha
        work[tail, j] = vector
        work[j, j] = alpha

        for k in I.parallel(cols[j + 1:N]):
            target = work[tail, k]
            projection = I.reduce.sum(vector * target, axis=0, acc_dtype=I.f32)
            work[tail, k] = target - (tau * projection) * vector

    # Apply the same reflectors to b, giving Q^T b. Each RHS is independent.
    for j in I.domain(0, N):
        tail = rows[j:M]
        stored = work[tail, j]
        coordinates = I.indices(tail)
        vector = I.select(coordinates == j, reflector_diagonal[j], stored)
        vector_sq = I.reduce.sum(vector * vector, axis=0, acc_dtype=I.f32)
        tau = I.cast(0.0, I.f32)
        if vector_sq > I.cast(0.0, I.f32):
            tau = I.fdiv(I.cast(2.0, I.f32), vector_sq)
        for rhs_col in I.parallel(rhs):
            target = y[tail, rhs_col]
            projection = I.reduce.sum(vector * target, axis=0, acc_dtype=I.f32)
            y[tail, rhs_col] = target - (tau * projection) * vector

    # Back substitution in R, with one ordered reverse sweep per RHS.
    for rhs_col in I.parallel(rhs):
        for ordinal in I.domain(0, N):
            i = N - 1 - ordinal
            suffix = cols[i + 1:N]
            upper = work[i, suffix]
            solved = x[suffix, rhs_col]
            correction = I.reduce.sum(upper * solved, axis=0, acc_dtype=I.f32)
            x[i, rhs_col] = I.fdiv(y[i, rhs_col] - correction, diagonal[i])


def build(context):
    artifact = context.compile("fused_qr_solve", qr_solve_kernel, constexprs={})

    def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = torch.empty((256, 10), device=A.device, dtype=A.dtype)
        artifact(A, b, x)
        return x

    return fused_qr_solve
