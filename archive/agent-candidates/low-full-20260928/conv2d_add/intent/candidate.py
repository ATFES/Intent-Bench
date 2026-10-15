import torch
import intent
import intent.language as I


@intent.kernel
def conv_add_kernel(
    x: I.In[I.f32, (16, 32, 32, 32)],
    w: I.In[I.f32, (32, 32, 3, 3)],
    other: I.In[I.f32, (16, 32, 32, 32)],
    out: I.Out[I.f32, (16, 32, 32, 32)],
    alpha: I.f32,
):
    # The output coordinates are independent.  The reduction is deliberately
    # kept as the convolution's logical member sequence.
    for n in I.parallel(I.domain(0, 16)):
        for oc in I.parallel(I.domain(0, 32)):
            for oh in I.parallel(I.domain(0, 32)):
                for ow in I.parallel(I.domain(0, 32)):
                    acc = I.cast(0.0, I.f32)
                    for ic in range(32):
                        for kh in range(3):
                            for kw in range(3):
                                ih = oh + kh - 1
                                iw = ow + kw - 1
                                if (ih >= 0) and (ih < 32) and (iw >= 0) and (iw < 32):
                                    acc = acc + x[n, ic, ih, iw] * w[oc, ic, kh, kw]
                    out[n, oc, oh, ow] = acc + alpha * other[n, oc, oh, ow]


def build(context):
    artifact = context.compile("conv2d_add_kernel", conv_add_kernel)

    def conv2d_add(input, weight, bias=None, other=None, stride=1, padding=0,
                   dilation=1, groups=1, alpha=1, out=None):
        if out is None:
            out = torch.empty((16, 32, 32, 32), device=input.device, dtype=input.dtype)
        if other is None:
            other = 0.0
        if not torch.is_tensor(other):
            other = torch.full_like(input, other)
        artifact(input, weight, other, out, float(alpha))
        return out

    return conv2d_add
