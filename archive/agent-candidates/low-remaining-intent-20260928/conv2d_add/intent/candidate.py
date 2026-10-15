import torch
import intent
import intent.language as I


@intent.kernel
def conv2d_add_kernel(
    inp: I.In[I.f32, (16, 32, 32, 32)],
    weight: I.In[I.f32, (32, 32, 3, 3)],
    other: I.In[I.f32, (16, 32, 32, 32)],
    out: I.Out[I.f32, (16, 32, 32, 32)],
    alpha: I.f32,
):
    ns = I.domain(0, 16)
    os = I.domain(0, 32)
    hs = I.domain(0, 32)
    ws = I.domain(0, 32)

    for n in I.parallel(ns):
        for oc in I.parallel(os):
            for oh in I.parallel(hs):
                for ow in I.parallel(ws):
                    acc = I.cast(0.0, I.f32)
                    for ic in I.domain(0, 32):
                        for kh in I.domain(0, 3):
                            for kw in I.domain(0, 3):
                                ih = oh + kh - 1
                                iw = ow + kw - 1
                                if (ih >= 0) and (ih < 32) and (iw >= 0) and (iw < 32):
                                    acc = acc + inp[n, ic, ih, iw] * weight[oc, ic, kh, kw]
                    out[n, oc, oh, ow] = acc + alpha * other[n, oc, oh, ow]


def build(context):
    kernel = context.compile("conv2d_add_kernel", conv2d_add_kernel)

    def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0,
                   dilation=1, groups=1, alpha=1, out=None):
        result = out if out is not None else torch.empty_like(input)
        kernel(input, weight, other, result, float(alpha))
        return result

    return conv2d_add
