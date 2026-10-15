import torch
import intent
import intent.language as I


@intent.kernel
def fused_kernel(
    input: I.In[I.f32, ("N", "M")],
    vec: I.In[I.f32, ("M",)],
    out: I.Out[I.f32, ("N",)],
    other: I.f32,
    alpha: I.f32,
):
    # Keep the contraction and pointwise epilogue in one launch.
    n, m = input.shape
    vec_col = I.reshape(vec, (m, 1))
    z_col = I.matmul(input, vec_col, acc_dtype=I.f32)
    z = I.reshape(z_col, (n,))
    activated = I.sigmoid(z)
    result = activated - alpha * other
    rows = I.domain(0, n)
    for row in I.parallel(rows):
        out[row] = result[row]


def build(context):
    compiled = context.compile("fused_mv_sigmoid_sub", fused_kernel)

    def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
        if out is None:
            out = torch.empty_like(vec)
        compiled(input, vec, out, other, alpha)
        return out

    return fused_mv_sigmoid_sub
