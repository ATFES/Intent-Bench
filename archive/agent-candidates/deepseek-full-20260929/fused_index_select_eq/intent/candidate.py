import torch
import intent
import intent.language as I


@intent.kernel
def _fused_index_select_eq_dim0(
    x: I.In[I.f32, ("M", "N")],
    index: I.In[I.i64, ("K",)],
    out: I.Out[I.bool, ("K", "N")],
    other: I.f32,
):
    M, N = x.shape
    K = index.shape[0]

    i = I.domain(0, K)
    j = I.domain(0, N)

    out[i, j] = (x[index[i], j] == other)


def build(context):
    kernel = context.compile("fused_index_select_eq_dim0", _fused_index_select_eq_dim0, constexprs={})

    def fused_index_select_eq(input, dim, index, other, *, out=None):
        K = index.shape[0]
        N = input.shape[1]
        if out is None:
            out = torch.empty((K, N), dtype=torch.bool, device=input.device)
        kernel(input, index, out, float(other))
        return out

    return fused_index_select_eq
