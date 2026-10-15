import torch
import intent
import intent.language as I


@intent.kernel
def _conv_bias(
    x: I.In[I.f32, (8, 64, 16, 16)],
    w: I.In[I.f32, (128, 64, 3, 3)],
    b: I.In[I.f32, (128,)],
    out: I.Out[I.f32, (8, 128, 14, 14)],
):
    cs = I.domain(0, 64)
    khs = I.domain(0, 3)
    kws = I.domain(0, 3)
    for n in I.parallel(I.domain(0, 8)):
        for oc in I.parallel(I.domain(0, 128)):
            for oh in I.parallel(I.domain(0, 14)):
                for ow in I.parallel(I.domain(0, 14)):
                    ci = I.reshape(I.indices(cs), (64, 1, 1))
                    hi = I.reshape(I.indices(khs), (1, 3, 1))
                    wi = I.reshape(I.indices(kws), (1, 1, 3))
                    xv = x[n, ci, oh + hi, ow + wi]
                    wv = w[oc, ci, hi, wi]
                    prod = xv * wv
                    acc = I.reduce.sum(prod, axis=(0, 1, 2), acc_dtype=I.f32)
                    out[n, oc, oh, ow] = acc + b[oc]


@intent.kernel
def _conv_nobias(
    x: I.In[I.f32, (8, 64, 16, 16)],
    w: I.In[I.f32, (128, 64, 3, 3)],
    out: I.Out[I.f32, (8, 128, 14, 14)],
):
    cs = I.domain(0, 64)
    khs = I.domain(0, 3)
    kws = I.domain(0, 3)
    for n in I.parallel(I.domain(0, 8)):
        for oc in I.parallel(I.domain(0, 128)):
            for oh in I.parallel(I.domain(0, 14)):
                for ow in I.parallel(I.domain(0, 14)):
                    ci = I.reshape(I.indices(cs), (64, 1, 1))
                    hi = I.reshape(I.indices(khs), (1, 3, 1))
                    wi = I.reshape(I.indices(kws), (1, 1, 3))
                    xv = x[n, ci, oh + hi, ow + wi]
                    wv = w[oc, ci, hi, wi]
                    out[n, oc, oh, ow] = I.reduce.sum(xv * wv, axis=(0, 1, 2), acc_dtype=I.f32)


def build(context):
    with_bias = context.compile("conv2d_bias", _conv_bias)
    without_bias = context.compile("conv2d_nobias", _conv_nobias)

    def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        n, _, ih, iw = input.shape
        oc, _, kh, kw = weight.shape
        if isinstance(stride, tuple):
            sh, sw = stride
        else:
            sh = sw = stride
        if isinstance(dilation, tuple):
            dh, dw = dilation
        else:
            dh = dw = dilation
        if padding == "same":
            ph = (kh - 1) * dh // 2
            pw = (kw - 1) * dw // 2
        elif padding == "valid":
            ph = pw = 0
        elif isinstance(padding, tuple):
            ph, pw = padding
        else:
            ph = pw = padding
        oh = (ih + 2 * ph - dh * (kh - 1) - 1) // sh + 1
        ow = (iw + 2 * pw - dw * (kw - 1) - 1) // sw + 1
        out = torch.empty((n, oc, oh, ow), device=input.device, dtype=input.dtype)
        if bias is None:
            without_bias(input, weight, out)
        else:
            with_bias(input, weight, bias, out)
        return out

    return conv2d
