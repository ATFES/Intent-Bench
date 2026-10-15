import torch
import intent
import intent.language as I


@intent.kernel
def _conv_relu_no_bias(
    input: I.In[I.f32, (1, 3, 256, 256)],
    weight: I.In[I.f32, (64, 3, 3, 3)],
    output: I.Out[I.f32, (1, 64, 256, 256)],
):
    for n in I.parallel(I.domain(0, 1)):
        for oc in I.parallel(I.domain(0, 64)):
            for oh in I.parallel(I.domain(0, 256)):
                for ow in I.parallel(I.domain(0, 256)):
                    acc = I.cast(0.0, I.f32)
                    for ic in range(3):
                        for kh in range(3):
                            for kw in range(3):
                                ih = oh + kh - 1
                                iw = ow + kw - 1
                                if ih >= 0 and ih < 256 and iw >= 0 and iw < 256:
                                    acc = acc + input[n, ic, ih, iw] * weight[oc, ic, kh, kw]
                    output[n, oc, oh, ow] = I.select(acc > 0.0, acc, I.cast(0.0, I.f32))


@intent.kernel
def _conv_relu_bias(
    input: I.In[I.f32, (1, 3, 256, 256)],
    weight: I.In[I.f32, (64, 3, 3, 3)],
    bias: I.In[I.f32, (64,)],
    output: I.Out[I.f32, (1, 64, 256, 256)],
):
    for n in I.parallel(I.domain(0, 1)):
        for oc in I.parallel(I.domain(0, 64)):
            for oh in I.parallel(I.domain(0, 256)):
                for ow in I.parallel(I.domain(0, 256)):
                    acc = bias[oc]
                    for ic in range(3):
                        for kh in range(3):
                            for kw in range(3):
                                ih = oh + kh - 1
                                iw = ow + kw - 1
                                if ih >= 0 and ih < 256 and iw >= 0 and iw < 256:
                                    acc = acc + input[n, ic, ih, iw] * weight[oc, ic, kh, kw]
                    output[n, oc, oh, ow] = I.select(acc > 0.0, acc, I.cast(0.0, I.f32))


def build(context):
    no_bias = context.compile("conv_relu_no_bias", _conv_relu_no_bias)
    with_bias = context.compile("conv_relu_bias", _conv_relu_bias)

    def relu_conv2d(input, weight, bias=None, stride=1, padding=0,
                    dilation=1, groups=1, inplace=False):
        if stride != 1 or padding != 1 or dilation != 1 or groups != 1:
            raise NotImplementedError("fixed profile only supports stride=1, padding=1, dilation=1, groups=1")
        output = torch.empty((1, 64, 256, 256), device=input.device, dtype=input.dtype)
        if bias is None:
            no_bias(input, weight, output)
        else:
            with_bias(input, weight, bias, output)
        return output

    return relu_conv2d
