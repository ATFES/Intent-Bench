import torch
import intent
import intent.language as I


@intent.kernel
def _fused_index_select_eq(
    input: I.In[I.f32, (8192, 128)],
    index: I.In[I.i64, (1024,)],
    other: I.f32,
    out: I.Out[I.bool, (1024, 128)],
):
    rows = I.domain(0, 1024)
    columns = I.domain(0, 128)
    selected = input[index[rows], columns]
    out[rows, columns] = selected == other


def build(context):
    artifact = context.compile(
        "fused_index_select_eq", _fused_index_select_eq, constexprs={}
    )

    def fused_index_select_eq(input, dim, index, other, *, out=None):
        if out is None:
            return artifact.run(input, index, other)
        artifact(input, index, other, out)
        return out

    return fused_index_select_eq
