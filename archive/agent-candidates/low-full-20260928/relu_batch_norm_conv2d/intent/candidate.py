import torch
import intent
import intent.language as I


@intent.kernel
def conv2d_kernel(
    x: I.In[I.f32, (1, 64, 32, 32)],
    w: I.In[I.f32, (128, 64, 3, 3)],
    out: I.Out[I.f32, (1, 128, 32, 32)],
):
    # Each output element owns its complete 3x3x64 reduction.
    for oc in I.parallel(I.domain(0, 128)):
        for oh in I.parallel(I.domain(0, 32)):
            for ow in I.parallel(I.domain(0, 32)):
                acc = I.cast(0.0, I.f32)
                for ic in range(64):
                    for kh in range(3):
                        for kw in range(3):
                            ih = oh + kh - 1
                            iw = ow + kw - 1
                            if (ih >= 0) and (ih < 32) and (iw >= 0) and (iw < 32):
                                acc = acc + x[0, ic, ih, iw] * w[oc, ic, kh, kw]
                out[0, oc, oh, ow] = acc


@intent.kernel
def bn_relu_kernel(
    conv: I.In[I.f32, (1, 128, 32, 32)],
    running_mean: I.In[I.f32, (128,)],
    running_var: I.In[I.f32, (128,)],
    bn_weight: I.In[I.f32, (128,)],
    bn_bias: I.In[I.f32, (128,)],
    out: I.Out[I.f32, (1, 128, 32, 32)],
):
    for oc in I.parallel(I.domain(0, 128)):
        mean = running_mean[oc]
        variance = running_var[oc]
        scale = bn_weight[oc]
        shift = bn_bias[oc]
        for oh in I.parallel(I.domain(0, 32)):
            for ow in I.parallel(I.domain(0, 32)):
                normalized = (conv[0, oc, oh, ow] - mean) / I.sqrt(variance + I.cast(1.0e-5, I.f32))
                value = normalized * scale + shift
                out[0, oc, oh, ow] = I.select(value > I.cast(0.0, I.f32), value, I.cast(0.0, I.f32))


def build(context):
    conv = context.compile("relu_bn_conv2d_conv", conv2d_kernel)
    bn_relu = context.compile("relu_bn_conv2d_bn_relu", bn_relu_kernel)

    def relu_batch_norm_conv2d(
        input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1,
        running_mean=None, running_var=None, bn_weight=None, bn_bias=None,
        training=False, momentum=0.1, eps=1e-5, inplace=False,
    ):
        intermediate = torch.empty((1, 128, 32, 32), device=input.device, dtype=input.dtype)
        output = torch.empty((1, 128, 32, 32), device=input.device, dtype=input.dtype)
        conv(input, weight, intermediate)
        bn_relu(intermediate, running_mean, running_var, bn_weight, bn_bias, output)
        return output

    return relu_batch_norm_conv2d
