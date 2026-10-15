import torch
import intent
import intent.language as I


@intent.kernel
def _conv(
    x: I.In[I.f32, ("N", "CI", "IH", "IW")],
    w: I.In[I.f32, ("CO", "GCI", "KH", "KW")],
    y: I.Out[I.f32, ("N", "CO", "OH", "OW")],
    stride_h: I.Constexpr[int],
    stride_w: I.Constexpr[int],
    pad_h: I.Constexpr[int],
    pad_w: I.Constexpr[int],
    dilation_h: I.Constexpr[int],
    dilation_w: I.Constexpr[int],
    groups: I.Constexpr[int],
):
    N, CI, IH, IW = x.shape
    CO, GCI, KH, KW = w.shape
    OH, OW = y.shape[2], y.shape[3]
    for n in I.parallel(I.domain(0, N)):
        for co in I.parallel(I.domain(0, CO)):
            for oh in I.parallel(I.domain(0, OH)):
                for ow in I.parallel(I.domain(0, OW)):
                    acc = I.cast(0.0, I.f32)
                    g = co // (CO // groups)
                    ci0 = g * GCI
                    for ci in I.domain(0, GCI):
                        for kh in I.domain(0, KH):
                            for kw in I.domain(0, KW):
                                ih = oh * stride_h - pad_h + kh * dilation_h
                                iw = ow * stride_w - pad_w + kw * dilation_w
                                valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                                xv = I.gather(x, (n, ci0 + ci, ih, iw), valid=valid, fill=I.cast(0.0, I.f32))
                                acc = acc + xv * w[co, ci, kh, kw]
                    y[n, co, oh, ow] = acc


@intent.kernel
def _conv_bias(
    x: I.In[I.f32, ("N", "CI", "IH", "IW")],
    w: I.In[I.f32, ("CO", "GCI", "KH", "KW")],
    b: I.In[I.f32, ("CO",)],
    y: I.Out[I.f32, ("N", "CO", "OH", "OW")],
    stride_h: I.Constexpr[int],
    stride_w: I.Constexpr[int],
    pad_h: I.Constexpr[int],
    pad_w: I.Constexpr[int],
    dilation_h: I.Constexpr[int],
    dilation_w: I.Constexpr[int],
    groups: I.Constexpr[int],
):
    N, CI, IH, IW = x.shape
    CO, GCI, KH, KW = w.shape
    OH, OW = y.shape[2], y.shape[3]
    for n in I.parallel(I.domain(0, N)):
        for co in I.parallel(I.domain(0, CO)):
            for oh in I.parallel(I.domain(0, OH)):
                for ow in I.parallel(I.domain(0, OW)):
                    acc = b[co]
                    g = co // (CO // groups)
                    ci0 = g * GCI
                    for ci in I.domain(0, GCI):
                        for kh in I.domain(0, KH):
                            for kw in I.domain(0, KW):
                                ih = oh * stride_h - pad_h + kh * dilation_h
                                iw = ow * stride_w - pad_w + kw * dilation_w
                                valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
                                xv = I.gather(x, (n, ci0 + ci, ih, iw), valid=valid, fill=I.cast(0.0, I.f32))
                                acc = acc + xv * w[co, ci, kh, kw]
                    y[n, co, oh, ow] = acc


@intent.kernel
def _gelu(
    x: I.In[I.f32, ("N", "C", "H", "W")],
    y: I.Out[I.f32, ("N", "C", "H", "W")],
    approximate: I.Constexpr[bool],
):
    N, C, H, W = x.shape
    inv_sqrt2 = I.cast(0.7071067811865475, I.f32)
    half = I.cast(0.5, I.f32)
    for n in I.parallel(I.domain(0, N)):
        for c in I.parallel(I.domain(0, C)):
            for h in I.parallel(I.domain(0, H)):
                for w in I.parallel(I.domain(0, W)):
                    v = x[n, c, h, w]
                    if approximate:
                        k = I.cast(0.7978845608028654, I.f32)
                        z = k * (v + I.cast(0.044715, I.f32) * v * v * v)
                        y[n, c, h, w] = half * v * (I.cast(1.0, I.f32) + I.tanh(z))
                    else:
                        y[n, c, h, w] = half * v * (I.cast(1.0, I.f32) + I.erf(v * inv_sqrt2))


def build(context):
    conv_args = {"stride_h": 1, "stride_w": 1, "pad_h": 1, "pad_w": 1,
                 "dilation_h": 1, "dilation_w": 1, "groups": 1}
    conv = context.compile("gelu_conv2d_conv", _conv, constexprs=conv_args)
    conv_bias = context.compile("gelu_conv2d_conv_bias", _conv_bias, constexprs=conv_args)
    gelu_exact = context.compile("gelu_conv2d_exact", _gelu, constexprs={"approximate": False})
    gelu_tanh = context.compile("gelu_conv2d_tanh", _gelu, constexprs={"approximate": True})

    def gelu_conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, approximate="none", out=None):
        sh, sw = (stride, stride) if isinstance(stride, int) else stride
        dh, dw = (dilation, dilation) if isinstance(dilation, int) else dilation
        if isinstance(padding, str):
            if padding == "same":
                ph = ((weight.shape[2] - 1) * dh) // 2
                pw = ((weight.shape[3] - 1) * dw) // 2
            else:
                ph = pw = 0
        else:
            ph, pw = (padding, padding) if isinstance(padding, int) else padding
        oh = (input.shape[2] + 2 * ph - dh * (weight.shape[2] - 1) - 1) // sh + 1
        ow = (input.shape[3] + 2 * pw - dw * (weight.shape[3] - 1) - 1) // sw + 1
        work = torch.empty((input.shape[0], weight.shape[0], oh, ow), device=input.device, dtype=input.dtype)
        if out is None:
            out = torch.empty_like(work)
        if bias is None:
            conv(input, weight, work)
        else:
            conv_bias(input, weight, bias, work)
        (gelu_tanh if approximate == "tanh" else gelu_exact)(work, out)
        return out

    return gelu_conv2d
