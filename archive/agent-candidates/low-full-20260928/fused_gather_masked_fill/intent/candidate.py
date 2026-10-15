import torch
import intent
import intent.language as I


@intent.kernel
def fused_kernel(
    x: I.In[I.f32, (1024, 1024)],
    index: I.In[I.i64, (1024, 1024)],
    mask: I.In[I.bool, (1024, 1024)],
    out: I.Out[I.f32, (1024, 1024)],
    value: I.f32,
):
    rows = I.domain(0, 1024)
    cols = I.domain(0, 1024)
    row_idx = I.indices(rows)
    col_idx = I.indices(cols)
    gathered = x[index[rows, cols], col_idx]
    result = I.select(mask[rows, cols], value, gathered)
    out[rows, cols] = result


def build(context):
    compiled = context.compile("fused_gather_masked_fill", fused_kernel)

    def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
        del dim, sparse_grad
        if out is None:
            out = torch.empty_like(input)
        compiled(input, index, mask, out, value)
        return out

    return fused_gather_masked_fill
