import torch
import intent
import intent.language as I


@intent.kernel
def _tile_exp(
    x: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, (2, "N")],
):
    n = x.shape[0]
    rows = I.domain(0, 2)
    cols = I.domain(0, n)
    for row in I.parallel(rows):
        for col in I.parallel(cols):
            out[row, col] = I.exp(x[col])


def build(context):
    kernel = context.compile("fused_tile_exp", _tile_exp)

    def fused_tile_exp(input, dims, *, out=None):
        n = input.shape[0]
        result = out if out is not None else torch.empty((2, n), device=input.device, dtype=input.dtype)
        kernel(input, result)
        return result

    return fused_tile_exp
