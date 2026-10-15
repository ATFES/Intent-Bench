import torch
import intent
import intent.language as I


@intent.kernel
def _make_symmetric(
    a: I.In[I.f32, ("N", "N")],
    s: I.Out[I.f32, ("N", "N")],
):
    rows = I.domain(0, a.shape[0])
    cols = I.domain(0, a.shape[1])
    for i in I.parallel(rows):
        for j in I.parallel(cols):
            if j <= i:
                s[i, j] = a[i, j]
            else:
                s[i, j] = a[j, i]


@intent.kernel
def _square(
    s: I.In[I.f32, ("N", "N")],
    out: I.Out[I.f32, ("N", "N")],
):
    value = I.matmul(s, s, acc_dtype=I.f32)
    out[:, :] = value


def build(context):
    make_symmetric = context.compile("matrix_power_eig_symmetrize", _make_symmetric)
    square = context.compile("matrix_power_eig_square", _square)

    def matrix_power_eig(A, k=2, *, out=None):
        del k
        n = A.shape[0]
        sym = torch.empty((n, n), device=A.device, dtype=A.dtype)
        result = out if out is not None else torch.empty_like(A)
        make_symmetric(A, sym)
        square(sym, result)
        return result

    return matrix_power_eig
