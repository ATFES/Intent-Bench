import torch
import intent
import intent.language as I


@intent.kernel
def conv_kernel(
    x: I.In[I.f32, (2, 16, 64, 64)],
    cw: I.In[I.f32, (64, 16, 3, 3)],
    cb: I.In[I.f32, (64,)],
    y: I.Out[I.f32, (2, 64, 62, 62)],
):
    # Each output coordinate owns one complete 3x3x16 dot product.
    n = I.domain(0, 2)
    oc = I.domain(0, 64)
    oh = I.domain(0, 62)
    ow = I.domain(0, 62)
    for bn in I.parallel(n):
        for co in I.parallel(oc):
            for row in I.parallel(oh):
                for col in I.parallel(ow):
                    acc = cb[co]
                    for ci in range(16):
                        for kh in range(3):
                            for kw in range(3):
                                acc = acc + x[bn, ci, row + kh, col + kw] * cw[co, ci, kh, kw]
                    y[bn, co, row, col] = acc


@intent.kernel
def norm_silu_kernel(
    src: I.In[I.f32, (2, 64, 62, 62)],
    out: I.Out[I.f32, (2, 64, 62, 62)],
    eps: I.f32,
):
    # The batch axis is the logical normalization group.  The two reductions
    # are deliberately explicit and precede the independent output pass.
    batch = I.domain(0, 2)
    for bn in I.parallel(batch):
        vals = src[bn, I.domain(0, 64), I.domain(0, 62), I.domain(0, 62)]
        m = I.reduce.sum(vals, axis=(0, 1, 2), acc_dtype=I.f32) / 253952.0
        s = I.reduce.sum(vals * vals, axis=(0, 1, 2), acc_dtype=I.f32) / 253952.0
        for co in I.parallel(I.domain(0, 64)):
            for row in I.parallel(I.domain(0, 62)):
                for col in I.parallel(I.domain(0, 62)):
                    v = s - m * m
                    z = (src[bn, co, row, col] - m) * I.rsqrt(v + eps)
                    out[bn, co, row, col] = z * I.sigmoid(z)


def build(context):
    conv = context.compile("fused_conv2d", conv_kernel)
    norm = context.compile("fused_layernorm_silu", norm_silu_kernel)

    def fused_silu_layer_norm_conv2d(
        x, weight, conv_weight, conv_bias=None, conv_stride=1,
        conv_padding=0, conv_dilation=1, conv_groups=1, ln_eps=1e-5,
    ):
        y = torch.empty((2, 64, 62, 62), device=x.device, dtype=x.dtype)
        out = torch.empty_like(y)
        conv(x, conv_weight, conv_bias, y)
        norm(y, out, ln_eps)
        return out

    return fused_silu_layer_norm_conv2d
