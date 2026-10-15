import torch
import intent
import intent.language as I


@intent.kernel
def _householder_qr(
    A: I.In[I.f32, ("N", "N")],
    R: I.Out[I.f32, ("N", "N")],
    parity: I.Out[I.f32, ()],
):
    n = A.shape[0]
    state = I.buffer((n, n), I.f32, init=A[:, :])
    flips = I.cast(1.0, I.f32)

    # Each iteration is one Householder reflector.  The outer loop is
    # deliberately ordered because later columns consume earlier updates.
    for k in I.domain(0, n):
        tail = I.domain(k, n)
        x = state[tail, k]
        norm2 = I.reduce.sum(x * x, axis=0, acc_dtype=I.f32)
        norm = I.sqrt(norm2)
        x0 = state[k, k]
        positive = x0 >= I.cast(0.0, I.f32)
        alpha = I.select(positive, -norm, norm)
        tail_indices = I.indices(tail)
        v = I.select(tail_indices == k, x - alpha, x)
        vv = I.reduce.sum(v * v, axis=0, acc_dtype=I.f32)
        active = vv > I.cast(0.0, I.f32)
        if active:
            beta = I.cast(2.0, I.f32) / vv
            for j in I.domain(k, n):
                col = state[tail, j]
                projection = I.reduce.sum(v * col, axis=0, acc_dtype=I.f32)
                state[tail, j] = col - v * (beta * projection)
            flips = -flips

    rows = I.domain(0, n)
    cols = I.domain(0, n)
    R[rows, cols] = state[rows, cols]
    parity[()] = flips


@intent.fn
def _mul(lhs: I.f32, rhs: I.f32) -> I.f32:
    return lhs * rhs


@intent.kernel
def _determinant_from_r(
    R: I.In[I.f32, ("N", "N")],
    parity: I.In[I.f32, ()],
    out: I.Out[I.f32, ()],
):
    n = R.shape[0]
    axis = I.domain(0, n)
    indices = I.indices(axis)
    diagonal = R[indices, indices]
    product = I.reduce(diagonal, axis=0, identity=I.cast(1.0, I.f32), combine=_mul)
    out[()] = product * parity[()]


def build(context):
    qr = context.compile("determinant_qr", _householder_qr)
    finish = context.compile("determinant_finish", _determinant_from_r)

    def determinant_via_qr(A, *, mode="reduced", out=None):
        n = A.shape[0]
        R = torch.empty((n, n), device=A.device, dtype=A.dtype)
        parity = torch.empty((), device=A.device, dtype=A.dtype)
        if out is None:
            out = torch.empty((), device=A.device, dtype=A.dtype)
        qr(A, R, parity)
        finish(R, parity, out)
        return out

    return determinant_via_qr
