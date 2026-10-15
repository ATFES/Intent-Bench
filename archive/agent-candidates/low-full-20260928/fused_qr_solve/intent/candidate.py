import torch
import intent
import intent.language as I


@intent.kernel
def _qr_kernel(
    A: I.In[I.f32, ("M", "N")],
    b: I.In[I.f32, ("M", "K")],
    R: I.Out[I.f32, ("M", "N")],
    Y: I.Out[I.f32, ("M", "K")],
):
    r = I.buffer((A.shape[0], A.shape[1]), I.f32, init=A[:, :])
    y = I.buffer((b.shape[0], b.shape[1]), I.f32, init=b[:, :])
    v = I.buffer((A.shape[0],), I.f32)
    m, n = A.shape
    kdim = b.shape[1]

    for j in range(n):
        jj = j
        ss = I.cast(0.0, I.f32)
        for i in range(j, m):
            q = r[i, jj]
            ss = ss + q * q
        norm = I.sqrt(ss)
        x0 = r[j, jj]
        sgn = I.cast(I.select(x0 >= I.cast(0.0, I.f32), 1.0, -1.0), I.f32)
        vj = x0 + sgn * norm
        v[j] = vj
        for i in range(j + 1, m):
            v[i] = r[i, jj]
        vv = I.cast(0.0, I.f32)
        for i in range(j, m):
            z = v[i]
            vv = vv + z * z
        beta = I.cast(2.0, I.f32) / vv
        for col in range(j, n):
            dot = I.cast(0.0, I.f32)
            for i in range(j, m):
                dot = dot + v[i] * r[i, col]
            scale = beta * dot
            for i in range(j, m):
                r[i, col] = r[i, col] - scale * v[i]
        for col in range(kdim):
            dot = I.cast(0.0, I.f32)
            for i in range(j, m):
                dot = dot + v[i] * y[i, col]
            scale = beta * dot
            for i in range(j, m):
                y[i, col] = y[i, col] - scale * v[i]

    for i in range(m):
        for col in range(n):
            R[i, col] = r[i, col]
        for col in range(kdim):
            Y[i, col] = y[i, col]


@intent.kernel
def _solve_kernel(
    R: I.In[I.f32, ("M", "N")],
    Y: I.In[I.f32, ("M", "K")],
    X: I.Out[I.f32, ("N", "K")],
):
    m, n = R.shape
    kdim = Y.shape[1]
    for col in range(kdim):
        for rev in range(n):
            i = n - 1 - rev
            rhs = Y[i, col]
            for p in range(i + 1, n):
                rhs = rhs - R[i, p] * X[p, col]
            X[i, col] = rhs / R[i, i]


def build(context):
    qr = context.compile("fused_qr_qt", _qr_kernel)
    solve = context.compile("fused_qr_backsolve", _solve_kernel)

    def fused_qr_solve(A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        m, n = A.shape
        k = b.shape[1]
        R = torch.empty((m, n), device=A.device, dtype=A.dtype)
        Y = torch.empty((m, k), device=b.device, dtype=b.dtype)
        X = torch.empty((n, k), device=A.device, dtype=A.dtype)
        qr(A, b, R, Y)
        solve(R, Y, X)
        return X

    return fused_qr_solve
