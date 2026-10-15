import torch
import intent
import intent.language as I


@intent.kernel
def _qr_factor(
    A: I.In[I.f32, (64, 16)],
    b: I.In[I.f32, (64, 1)],
    R: I.Out[I.f32, (16, 16)],
    qtb: I.Out[I.f32, (16, 1)],
):
    rows = I.domain(0, 64)
    cols = I.domain(0, 16)
    one = I.domain(0, 1)
    zero = I.cast(0.0, I.f32)
    two = I.cast(2.0, I.f32)

    r = I.buffer((64, 16), I.f32, init=A[rows, cols])
    y = I.buffer((64, 1), I.f32, init=b[rows, one])

    # Each reflector is an ordered dependency; its trailing column updates are
    # independent once the reflector vector has been formed.
    for j in cols:
        tail = rows[j:64]
        x = r[tail, j]
        x0 = r[j, j]
        norm = I.sqrt(I.reduce.sum(x * x, axis=0, acc_dtype=I.f32))
        alpha = I.select(x0 >= zero, -norm, norm)
        v0 = x0 - alpha
        tail_indices = I.indices(tail)
        v = I.select(tail_indices == j, v0, x)
        vv = I.reduce.sum(v * v, axis=0, acc_dtype=I.f32)
        beta = two / vv

        for k in I.parallel(cols[j:16]):
            column = r[tail, k]
            projection = I.reduce.sum(v * column, axis=0, acc_dtype=I.f32)
            r[tail, k] = column - (beta * projection) * v

        y_tail = y[tail, 0]
        y_projection = I.reduce.sum(v * y_tail, axis=0, acc_dtype=I.f32)
        y[tail, 0] = y_tail - (beta * y_projection) * v
        r[j, j] = alpha

    for row in I.parallel(I.domain(0, 16)):
        R[row, cols] = r[row, cols]
        qtb[row, one] = y[row, one]


@intent.kernel
def _back_substitute(
    R: I.In[I.f32, (16, 16)],
    qtb: I.In[I.f32, (16, 1)],
    x: I.Out[I.f32, (16, 1)],
):
    # Reverse order is required because each equation depends on solved rows.
    for ordinal in I.domain(0, 16):
        i = 15 - ordinal
        rhs = qtb[i, 0]
        for k in I.domain(i + 1, 16):
            rhs = rhs - R[i, k] * x[k, 0]
        x[i, 0] = rhs / R[i, i]


def build(context):
    qr_artifact = context.compile("least_squares_qr_factor", _qr_factor)
    solve_artifact = context.compile("least_squares_qr_solve", _back_substitute)

    def least_squares_qr(A, b, *, mode="reduced", out=None):
        del mode
        if out is None:
            out = torch.empty((16, 1), device=A.device, dtype=A.dtype)
        R = torch.empty((16, 16), device=A.device, dtype=A.dtype)
        qtb = torch.empty((16, 1), device=A.device, dtype=A.dtype)
        qr_artifact(A, b, R, qtb)
        solve_artifact(R, qtb, out)
        return out

    return least_squares_qr
