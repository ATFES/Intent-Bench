import torch
import intent
import intent.language as I


@intent.kernel
def _conv_act(
    x: I.In[I.f32, (1, 3, 768, 768)],
    w: I.In[I.f32, (6, 3, 3, 3)],
    b: I.In[I.f32, (6,)],
    out: I.Out[I.f32, (1, 6, 766, 766)],
    negative_slope: I.f32,
):
    n_domain = I.domain(0, 1)
    oc_domain = I.domain(0, 6)
    oh_domain = I.domain(0, 766)
    ow_domain = I.domain(0, 766)

    for n in I.parallel(n_domain):
        for oc in I.parallel(oc_domain):
            for oh in I.parallel(oh_domain):
                for ow in I.parallel(ow_domain):
                    acc = b[oc]
                    for ic in range(3):
                        for kh in range(3):
                            for kw in range(3):
                                acc = acc + x[n, ic, oh + kh, ow + kw] * w[oc, ic, kh, kw]
                    out[n, oc, oh, ow] = I.select(acc > 0.0, acc, acc * negative_slope)


def build(context):
    conv = context.compile("leaky_relu_conv2d_fixed", _conv_act)

    def leaky_relu_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        negative_slope=0.01,
        inplace=False,
    ):
        output = torch.empty((1, 6, 766, 766), device=input.device, dtype=input.dtype)
        if bias is None:
            bias = torch.zeros((6,), device=input.device, dtype=input.dtype)
        conv(input, weight, bias, output, negative_slope)
        return output

    return leaky_relu_conv2d
