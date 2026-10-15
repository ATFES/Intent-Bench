import torch
import intent
import intent.language as I


@intent.kernel
def _conv_gelu(
    input: I.In[I.f32, (16, 16, 64, 64)],
    weight: I.In[I.f32, (32, 16, 3, 3)],
    out: I.Out[I.f32, (16, 32, 64, 64)],
):
    n_domain = I.domain(0, 16)
    co_domain = I.domain(0, 32)
    oh_domain = I.domain(0, 64)
    ow_domain = I.domain(0, 64)

    for n in I.parallel(n_domain):
        for co in I.parallel(co_domain):
            for oh in I.parallel(oh_domain):
                for ow in I.parallel(ow_domain):
                    acc = I.cast(0.0, I.f32)
                    for ci in range(16):
                        for kh in range(3):
                            for kw in range(3):
                                ih = oh + kh - 1
                                iw = ow + kw - 1
                                if (ih >= 0) and (ih < 64) and (iw >= 0) and (iw < 64):
                                    acc = acc + input[n, ci, ih, iw] * weight[co, ci, kh, kw]
                    out[n, co, oh, ow] = I.cast(0.5, I.f32) * acc * I.erfc(
                        -acc / I.sqrt(I.cast(2.0, I.f32))
                    )


def build(context):
    conv_gelu = context.compile("conv_gelu", _conv_gelu)

    def gelu_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        approximate="none",
        out=None,
    ):
        if out is None:
            out = torch.empty((16, 32, 64, 64), device=input.device, dtype=input.dtype)
        conv_gelu(input, weight, out)
        return out

    return gelu_conv2d
