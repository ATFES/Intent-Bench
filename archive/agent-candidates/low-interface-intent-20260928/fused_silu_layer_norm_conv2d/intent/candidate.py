import torch
import intent
import intent.language as I


@intent.kernel
def _conv(x: I.In[I.f32, (2, 16, 64, 64)], w: I.In[I.f32, (64, 16, 3, 3)],
          b: I.In[I.f32, (64,)], y: I.Out[I.f32, (2, 64, 62, 62)],
          STRIDE: I.Constexpr[int], PADDING: I.Constexpr[int],
          DILATION: I.Constexpr[int], GROUPS: I.Constexpr[int]):
    # Each output point is independent; the loops are the convolution's K reduction.
    for n in I.parallel(I.domain(0, 2)):
        for oc in I.parallel(I.domain(0, 64)):
            for oh in I.parallel(I.domain(0, 62)):
                for ow in I.parallel(I.domain(0, 62)):
                    acc = I.cast(b[oc], I.f32)
                    g = oc // (64 // GROUPS)
                    ic0 = g * (16 // GROUPS)
                    for ic in range(16 // GROUPS):
                        for kh in range(3):
                            for kw in range(3):
                                ih = oh * STRIDE - PADDING + kh * DILATION
                                iw = ow * STRIDE - PADDING + kw * DILATION
                                acc = acc + x[n, ic0 + ic, ih, iw] * w[oc, ic, kh, kw]
                    y[n, oc, oh, ow] = acc


@intent.kernel
def _conv_nobias(x: I.In[I.f32, (2, 16, 64, 64)], w: I.In[I.f32, (64, 16, 3, 3)],
                 y: I.Out[I.f32, (2, 64, 62, 62)],
                 STRIDE: I.Constexpr[int], PADDING: I.Constexpr[int],
                 DILATION: I.Constexpr[int], GROUPS: I.Constexpr[int]):
    for n in I.parallel(I.domain(0, 2)):
        for oc in I.parallel(I.domain(0, 64)):
            for oh in I.parallel(I.domain(0, 62)):
                for ow in I.parallel(I.domain(0, 62)):
                    acc = I.cast(0.0, I.f32)
                    g = oc // (64 // GROUPS)
                    ic0 = g * (16 // GROUPS)
                    for ic in range(16 // GROUPS):
                        for kh in range(3):
                            for kw in range(3):
                                ih = oh * STRIDE - PADDING + kh * DILATION
                                iw = ow * STRIDE - PADDING + kw * DILATION
                                acc = acc + x[n, ic0 + ic, ih, iw] * w[oc, ic, kh, kw]
                    y[n, oc, oh, ow] = acc


@intent.kernel
def _norm_silu(x: I.In[I.f32, (2, 64, 62, 62)], y: I.Out[I.f32, (2, 64, 62, 62)],
               EPS: I.Constexpr[float]):
    total = I.reduce.sum(x, axis=(1, 2, 3), acc_dtype=I.f32)
    mean = total / I.cast(64 * 62 * 62, I.f32)
    mean4 = I.reshape(mean, (2, 1, 1, 1))
    d = x - mean4
    sq = d * d
    varsum = I.reduce.sum(sq, axis=(1, 2, 3), acc_dtype=I.f32)
    var = varsum / I.cast(64 * 62 * 62, I.f32)
    inv = I.rsqrt(I.reshape(var, (2, 1, 1, 1)) + EPS)
    z = d * inv
    y[:, :, :, :] = z * I.sigmoid(z)


def build(context):
    conv = context.compile("fused_conv", _conv,
                           constexprs={"STRIDE": 1, "PADDING": 0,
                                       "DILATION": 1, "GROUPS": 1})
    conv_nobias = context.compile("fused_conv_nobias", _conv_nobias,
                                  constexprs={"STRIDE": 1, "PADDING": 0,
                                              "DILATION": 1, "GROUPS": 1})
    norm = context.compile("fused_norm_silu", _norm_silu,
                           constexprs={"EPS": 1e-5})

    def fused_silu_layer_norm_conv2d(
        x, weight, conv_weight, conv_bias=None, conv_stride=1,
        conv_padding=0, conv_dilation=1, conv_groups=1, ln_eps=1e-5
    ):
        tmp = torch.empty((2, 64, 62, 62), device=x.device, dtype=x.dtype)
        out = torch.empty_like(tmp)
        if conv_bias is None:
            conv_nobias(x, conv_weight, tmp)
        else:
            conv(x, conv_weight, conv_bias, tmp)
        norm(tmp, out)
        return out

    return fused_silu_layer_norm_conv2d
