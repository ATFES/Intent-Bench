import torch
import intent
import intent.language as I


@intent.kernel
def least_squares_qr_kernel(
    A: I.In[I.f32, ("M", "N")],
    B: I.In[I.f32, ("M", "K")],
    W: I.InOut[I.f32, ("M", "N")],
    C: I.InOut[I.f32, ("M", "K")],
    X: I.InOut[I.f32, ("N", "K")],
):
    M, N = A.shape
    K = B.shape[1]

    rows = I.domain(0, M)
    cols = I.domain(0, N)
    kcols = I.domain(0, K)

    W[rows, cols] = A[rows, cols]
    C[rows, kcols] = B[rows, kcols]

    zero = I.cast(0.0, I.f32)
    two = I.cast(2.0, I.f32)

    # Householder QR: A = Q R, with R stored in the upper triangle of W and
    # Q^T B accumulated in C.
    for j in I.domain(0, N):
        tail = rows[j:M]
        col = W[tail, j]

        norm = I.sqrt(I.reduce.sum(col * col, axis=0, acc_dtype=I.f32))
        x0 = W[j, j]
        alpha = I.select(x0 >= zero, -norm, norm)

        coords = I.indices(tail)
        v = I.select(coords == j, col - alpha, col)

        vnorm_sq = I.reduce.sum(v * v, axis=0, acc_dtype=I.f32)
        beta = I.select(vnorm_sq > zero, two / vnorm_sq, zero)

        if j + 1 < N:
            tail_cols = cols[j + 1 : N]
            block = W[tail, tail_cols]
            proj = I.vecmat(v, block, acc_dtype=I.f32)
            W[tail, tail_cols] = block - beta * I.outer(v, proj)

        proj_b = I.vecmat(v, C[tail, kcols], acc_dtype=I.f32)
        C[tail, kcols] = C[tail, kcols] - beta * I.outer(v, proj_b)

        W[j, j] = alpha

    # Solve R x = C[:N, :] by back substitution.
    for ordinal in I.domain(0, N):
        i = N - 1 - ordinal
        if i + 1 < N:
            row_upper = W[i, cols[i + 1 : N]]
            x_lower = X[cols[i + 1 : N], kcols]
            acc = I.vecmat(row_upper, x_lower, acc_dtype=I.f32)
            rhs = C[i, kcols] - acc
        else:
            rhs = C[i, kcols]
        X[i, kcols] = rhs / W[i, i]


def build(context):
    qr = context.compile("least_squares_qr_kernel", least_squares_qr_kernel)

    def least_squares_qr(A, b, *, mode="reduced", out=None):
        m, n = A.shape
        b_is_matrix = b.dim() == 2
        b_matrix = b if b_is_matrix else b.unsqueeze(-1)
        k = b_matrix.shape[1]

        if out is None:
            if b_is_matrix:
                x = torch.empty((n, k), dtype=A.dtype, device=A.device)
            else:
                x = torch.empty((n,), dtype=A.dtype, device=A.device)
        else:
            x = out

        w = torch.empty((m, n), dtype=A.dtype, device=A.device)
        c = torch.empty((m, k), dtype=A.dtype, device=A.device)

        x_view = x if b_is_matrix else x.unsqueeze(-1)
        qr(A, b_matrix, w, c, x_view)

        return x

    return least_squares_qr
