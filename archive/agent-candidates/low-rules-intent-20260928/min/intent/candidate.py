import torch
import intent
import intent.language as I


@intent.fn
def _min_first(lhs, rhs):
    lv, li = lhs
    rv, ri = rhs
    take_left = (lv < rv) | ((lv == rv) & (li <= ri))
    return (I.select(take_left, lv, rv), I.select(take_left, li, ri))


@intent.kernel
def _min_dim0(
    x: I.In[I.f32, ("M", "N")],
    out_values: I.Out[I.f32, ("N",)],
    out_indices: I.Out[I.i64, ("N",)],
):
    rows = I.domain(0, x.shape[0])
    cols = I.domain(0, x.shape[1])
    values = x[rows, cols]
    row_indices = I.reshape(I.indices(rows), (x.shape[0], 1))
    indices = row_indices + I.full((1, x.shape[1]), 0, dtype=I.index)
    summary = I.reduce(
        (values, indices),
        axis=0,
        identity=(I.cast(I.inf, I.f32), I.cast(0, I.index)),
        combine=_min_first,
    )
    out_values[cols] = summary[0]
    out_indices[cols] = I.cast(summary[1], I.i64)


def build(context):
    kernel = context.compile("min_dim0", _min_dim0)

    def min(input_tensor, dim, keepdim=False):
        if dim != 0:
            raise ValueError("fixed profile supports dim=0")
        if keepdim:
            raise ValueError("fixed profile supports keepdim=False")
        n = input_tensor.shape[1]
        values = torch.empty((n,), device=input_tensor.device, dtype=input_tensor.dtype)
        indices = torch.empty((n,), device=input_tensor.device, dtype=torch.int64)
        kernel(input_tensor, values, indices)
        return values, indices

    return min
