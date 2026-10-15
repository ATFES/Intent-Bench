import torch
import intent
import intent.language as I


@intent.fn
def _min_pair(lhs, rhs):
    take_lhs = (lhs[0] < rhs[0]) | ((lhs[0] == rhs[0]) & (lhs[1] <= rhs[1]))
    return (I.select(take_lhs, lhs[0], rhs[0]), I.select(take_lhs, lhs[1], rhs[1]))


@intent.kernel
def _min_rows(
    x: I.In[I.f32, ("M", "N")],
    out: I.Out[I.f32, ("N",)],
    ind: I.Out[I.i64, ("N",)],
):
    M, N = x.shape
    rows = I.domain(0, M)
    row_index = I.reshape(I.indices(rows), (M, 1))
    indices = I.cast(row_index, I.i64) + I.full((M, 1), 0, dtype=I.i64)
    values = x[rows, I.domain(0, N)]
    pair = (values, indices)
    identity = (I.full((N,), I.inf, dtype=I.f32), I.full((N,), M, dtype=I.i64))
    result = I.reduce(pair, axis=0, identity=identity, combine=_min_pair)
    out[I.domain(0, N)] = result[0]
    ind[I.domain(0, N)] = result[1]


@intent.kernel
def _min_rows_keepdim(
    x: I.In[I.f32, ("M", "N")],
    out: I.Out[I.f32, (1, "N")],
    ind: I.Out[I.i64, (1, "N")],
):
    M, N = x.shape
    rows = I.domain(0, M)
    row_index = I.reshape(I.indices(rows), (M, 1))
    indices = I.cast(row_index, I.i64) + I.full((M, 1), 0, dtype=I.i64)
    values = x[rows, I.domain(0, N)]
    pair = (values, indices)
    identity = (I.full((N,), I.inf, dtype=I.f32), I.full((N,), M, dtype=I.i64))
    result = I.reduce(pair, axis=0, identity=identity, combine=_min_pair)
    out[I.domain(0, 1), I.domain(0, N)] = I.reshape(result[0], (1, N))
    ind[I.domain(0, 1), I.domain(0, N)] = I.reshape(result[1], (1, N))


def build(context):
    normal = context.compile("min_rows", _min_rows)
    keep = context.compile("min_rows_keepdim", _min_rows_keepdim)

    def min(input_tensor, dim, keepdim=False):
        if dim != 0:
            raise ValueError("fixed profile supports dim=0")
        _, n = input_tensor.shape
        shape = (1, n) if keepdim else (n,)
        values = torch.empty(shape, device=input_tensor.device, dtype=input_tensor.dtype)
        indices = torch.empty(shape, device=input_tensor.device, dtype=torch.int64)
        if keepdim:
            keep(input_tensor, values, indices)
        else:
            normal(input_tensor, values, indices)
        return values, indices

    return min
