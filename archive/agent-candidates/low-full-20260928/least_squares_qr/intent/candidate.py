import torch
import intent
import intent.language as I


@intent.kernel
def _qr_apply(
    A: I.In[I.f32, (64, 16)],
    b: I.In[I.f32, (64, 1)],
    R: I.Out[I.f32, (64, 16)],
    qtb: I.Out[I.f32, (64, 1)],
):
    # Copy the inputs into logical mutable state.  The ordered outer loop is
    # the Householder dependency; rows/columns inside each step are disjoint.
    for i in range(64):
        for j in range(16):
            R[i, j] = A[i, j]
    for i in range(64):
        qtb[i, 0] = b[i, 0]

    for j in range(16):
        norm2 = I.cast(0.0, I.f32)
        for i in range(j, 64):
            t = R[i, j]
            norm2 = norm2 + t * t
        norm = I.sqrt(norm2)
        x0 = R[j, j]
        s = I.select(x0 >= I.cast(0.0, I.f32), I.cast(1.0, I.f32), I.cast(-1.0, I.f32))
        alpha = -s * norm
        R[j, j] = x0 - alpha
        vv = I.cast(0.0, I.f32)
        for i in range(j, 64):
            t = R[i, j]
            vv = vv + t * t
        beta = I.cast(2.0, I.f32) / vv
        for k in range(j + 1, 16):
            dot = I.cast(0.0, I.f32)
            for i in range(j, 64):
                dot = dot + R[i, j] * R[i, k]
            scale = beta * dot
            for i in range(j, 64):
                R[i, k] = R[i, k] - scale * R[i, j]
        dotb = I.cast(0.0, I.f32)
        for i in range(j, 64):
            dotb = dotb + R[i, j] * qtb[i, 0]
        scale_b = beta * dotb
        for i in range(j, 64):
            qtb[i, 0] = qtb[i, 0] - scale_b * R[i, j]
        R[j, j] = alpha


@intent.kernel
def _backsolve(
    R: I.In[I.f32, (64, 16)],
    qtb: I.In[I.f32, (64, 1)],
    out: I.Out[I.f32, (16, 1)],
):
    for ii in range(16):
        i = 15 - ii
        value = qtb[i, 0]
        for j in range(i + 1, 16):
            value = value - R[i, j] * out[j, 0]
        out[i, 0] = value / R[i, i]


def build(context):
    qr = context.compile("least_squares_qr_householder", _qr_apply)
    solve = context.compile("least_squares_qr_backsolve", _backsolve)

    def least_squares_qr(A, b, *, mode="reduced", out=None):
        R = torch.empty((64, 16), device=A.device, dtype=A.dtype)
        qtb = torch.empty((64, 1), device=A.device, dtype=A.dtype)
        result = out if out is not None else torch.empty((16, 1), device=A.device, dtype=A.dtype)
        qr(A, b, R, qtb)
        solve(R, qtb, result)
        return result

    return least_squares_qr
