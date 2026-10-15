from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def _conv(
    x: I.In[I.f32, ("N", "CI", "IH", "IW")],
    w: I.In[I.f32, ("CO", "CI", "KH", "KW")],
    b: I.In[I.f32, ("CO",)],
    y: I.Out[I.f32, ("N", "CO", "OH", "OW")],
    HAS_BIAS: I.Constexpr[bool],
):
    N, CI, IH, IW = x.shape
    CO, _, KH, KW = w.shape
    OH, OW = y.shape[2], y.shape[3]
    channels = I.domain(0, CI)
    rows = I.domain(0, KH)
    cols = I.domain(0, KW)
    for n in I.parallel(I.domain(0, N)):
        for oc in I.parallel(I.domain(0, CO)):
            for oh in I.parallel(I.domain(0, OH)):
                for ow in I.parallel(I.domain(0, OW)):
                    ci = I.reshape(I.indices(channels), (CI, 1, 1))
                    kh = I.indices(rows)
                    kw = I.indices(cols)
                    ih = I.reshape(oh + kh, (1, KH, 1))
                    iw = I.reshape(ow + kw, (1, 1, KW))
                    xv = x[n, ci, ih, iw]
                    wv = w[oc, ci, kh[:, None], kw[None, :]]
                    # The channel and two kernel axes are the complete
                    # logical reduction for each independent output point.
                    acc = I.reduce.sum(xv * wv, axis=(0, 1, 2), acc_dtype=I.f32)
                    if HAS_BIAS:
                        acc = acc + b[oc]
                    y[n, oc, oh, ow] = acc


@intent.kernel
def _sigmoid(
    src: I.In[I.f32, ("N", "CO", "OH", "OW")],
    dst: I.Out[I.f32, ("N", "CO", "OH", "OW")],
):
    N, CO, OH, OW = src.shape
    for n in I.parallel(I.domain(0, N)):
        for co in I.parallel(I.domain(0, CO)):
            for oh in I.parallel(I.domain(0, OH)):
                for ow in I.parallel(I.domain(0, OW)):
                    v = src[n, co, oh, ow]
                    dst[n, co, oh, ow] = I.fdiv(
                        I.cast(1.0, I.f32),
                        I.cast(1.0, I.f32) + I.exp(-v),
                    )


def build(context):
    conv_bias = context.compile(
        "sigmoid_conv2d_conv_bias", _conv, constexprs={"HAS_BIAS": True}
    )
    conv_no_bias = context.compile(
        "sigmoid_conv2d_conv_no_bias", _conv, constexprs={"HAS_BIAS": False}
    )
    sigmoid = context.compile("sigmoid_conv2d_sigmoid", _sigmoid)

    def sigmoid_conv2d(
        input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, out=None
    ):
        if isinstance(stride, tuple):
            stride_h, stride_w = stride
        else:
            stride_h = stride_w = stride
        if isinstance(dilation, tuple):
            dilation_h, dilation_w = dilation
        else:
            dilation_h = dilation_w = dilation
        if isinstance(padding, str):
            if padding == "valid":
                pad_h = pad_w = 0
            elif padding == "same":
                pad_h = (weight.shape[2] - 1) * dilation_h // 2
                pad_w = (weight.shape[3] - 1) * dilation_w // 2
            else:
                raise ValueError("unsupported padding")
        elif isinstance(padding, tuple):
            pad_h, pad_w = padding
        else:
            pad_h = pad_w = padding
        n, ci, ih, iw = input.shape
        co, _, kh, kw = weight.shape
        oh = (ih + 2 * pad_h - dilation_h * (kh - 1) - 1) // stride_h + 1
        ow = (iw + 2 * pad_w - dilation_w * (kw - 1) - 1) // stride_w + 1
        if groups != 1 or stride_h != 1 or stride_w != 1 or pad_h != 0 or pad_w != 0 or dilation_h != 1 or dilation_w != 1:
            raise ValueError("fixed profile only supports the documented default convolution")
        if out is None:
            conv = torch.empty((n, co, oh, ow), device=input.device, dtype=input.dtype)
            result = torch.empty_like(conv)
        else:
            conv = torch.empty((n, co, oh, ow), device=input.device, dtype=input.dtype)
            result = out
        if bias is None:
            # The no-bias specialization does not read this view; it is only
            # present because the shared kernel interface keeps argument order.
            unused_bias = torch.empty((co,), device=input.device, dtype=input.dtype)
            conv_no_bias(input, weight, unused_bias, conv)
        else:
            conv_bias(input, weight, bias, conv)
        sigmoid(conv, result)
        return result

    return sigmoid_conv2d
