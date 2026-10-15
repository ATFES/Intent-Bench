import torch
import intent
import intent.language as I


@intent.kernel
def conv2d_kernel(
    input: I.In[I.f32, (8, 64, 16, 16)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    bias: I.In[I.f32, (128,)],
    output: I.Out[I.f32, (8, 128, 14, 14)],
):
    for n in I.parallel(I.domain(0, 8)):
        for oc in I.parallel(I.domain(0, 128)):
            for oh in I.parallel(I.domain(0, 14)):
                for ow in I.parallel(I.domain(0, 14)):
                    accum = I.cast(0.0, I.f32)
                    for ic in I.domain(0, 64):
                        for kh in range(3):
                            for kw in range(3):
                                ih = oh + kh
                                iw = ow + kw
                                if (ih < 16) and (iw < 16):
                                    accum = accum + input[n, ic, ih, iw] * weight[oc, ic, kh, kw]
                    accum = accum + bias[oc]
                    output[n, oc, oh, ow] = accum


def build(context):
    artifact = context.compile("conv2d", conv2d_kernel)

    def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        if bias is None:
            bias = torch.zeros(weight.shape[0], dtype=input.dtype, device=input.device)
        output = torch.empty(input.shape[0], weight.shape[0], 14, 14, dtype=input.dtype, device=input.device)
        artifact(input, weight, bias, output)
        return output

    return conv2d

