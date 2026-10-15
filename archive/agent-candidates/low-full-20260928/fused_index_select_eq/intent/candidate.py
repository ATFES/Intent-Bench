import torch
import intent
import intent.language as I


@intent.kernel
def _fused_index_select_eq(
    input: I.In[I.f32, ("M", "N")],
    index: I.In[I.i64, ("K",)],
    other: I.f32,
    out: I.Out[I.bool, ("K", "N")],
):
    rows = I.domain(0, index.shape[0])
    cols = I.domain(0, input.shape[1])
    selected = input[index[rows], cols]
    out[rows, cols] = selected == other


def build(context):
    kernel = context.compile("fused_index_select_eq", _fused_index_select_eq)

    def fused_index_select_eq(input, dim, index, other, *, out=None):
        result = out
        if result is None:
            result = torch.empty(
                (index.shape[0], input.shape[1]),
                dtype=torch.bool,
                device=input.device,
            )
        kernel(input, index, other, result)
        return result

    return fused_index_select_eq
