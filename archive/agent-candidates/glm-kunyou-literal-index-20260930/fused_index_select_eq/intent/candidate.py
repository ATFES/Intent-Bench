from __future__ import annotations

import torch
import intent
import intent.language as I


# Fixed profile: input (8192, 128) f32, dim=0, index (1024,) i64, other scalar f32,
# result (1024, 128) bool.
# One kernel: I.gather selects rows of X along dim 0 (index tensor has shape (1024,),
# source has shape (8192, 128), result has shape (1024, 128)).  Then element-wise
# equality against the scalar Y broadcasts implicitly and produces the boolean output.
@intent.kernel
def _index_select_eq(
    x: I.In[I.f32, (8192, 128)],
    idx: I.In[I.i64, (1024,)],
    other: I.f32,
    out: I.Out[I.bool, (1024, 128)],
):
    selected = I.gather(x, idx)
    out[:, :] = selected == other


def build(context):
    kernel = context.compile("fused_index_select_eq", _index_select_eq)

    def fused_index_select_eq(input, dim, index, other, *, out=None):
        if out is None:
            result_shape = (
                tuple(input.shape[:dim]) + tuple(index.shape) + tuple(input.shape[dim + 1:])
            )
            out = torch.empty(result_shape, dtype=torch.bool, device=input.device)
        kernel(input, index, float(other), out)
        return out

    return fused_index_select_eq

