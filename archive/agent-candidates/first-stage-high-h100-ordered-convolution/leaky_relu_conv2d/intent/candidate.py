import torch
import intent
import intent.language as I


@intent.kernel
def _conv2d_leaky(
    input: I.In[I.f32, (1, 3, 768, 768)],
    weight: I.In[I.f32, (6, 3, 3, 3)],
    bias: I.In[I.f32, (6,)],
    output: I.Out[I.f32, (1, 6, 766, 766)],
    negative_slope: I.f32,
):
    for n in I.parallel(I.domain(0, 1)):
        for oc in I.parallel(I.domain(0, 6)):
            for oh in I.parallel(I.domain(0, 766)):
                for ow in I.parallel(I.domain(0, 766)):
                    channels = I.domain(0, 3)
                    kernel_rows = I.domain(0, 3)
                    kernel_cols = I.domain(0, 3)
                    channel_idx = I.reshape(I.indices(channels), (3, 1, 1))
                    kernel_row_idx = I.reshape(I.indices(kernel_rows), (1, 3, 1))
                    kernel_col_idx = I.reshape(I.indices(kernel_cols), (1, 1, 3))
                    products = (
                        input[n, channel_idx, oh + kernel_row_idx, ow + kernel_col_idx]
                        * weight[oc, channel_idx, kernel_row_idx, kernel_col_idx]
                    )
                    value = bias[oc] + I.reduce.sum(
                        products,
                        axis=(0, 1, 2),
                        acc_dtype=I.f32,
                    )
                    output[n, oc, oh, ow] = I.select(
                        value > 0.0,
                        value,
                        value * negative_slope,
                    )


@intent.kernel
def _conv2d_leaky_no_bias(
    input: I.In[I.f32, (1, 3, 768, 768)],
    weight: I.In[I.f32, (6, 3, 3, 3)],
    output: I.Out[I.f32, (1, 6, 766, 766)],
    negative_slope: I.f32,
):
    for n in I.parallel(I.domain(0, 1)):
        for oc in I.parallel(I.domain(0, 6)):
            for oh in I.parallel(I.domain(0, 766)):
                for ow in I.parallel(I.domain(0, 766)):
                    channels = I.domain(0, 3)
                    kernel_rows = I.domain(0, 3)
                    kernel_cols = I.domain(0, 3)
                    channel_idx = I.reshape(I.indices(channels), (3, 1, 1))
                    kernel_row_idx = I.reshape(I.indices(kernel_rows), (1, 3, 1))
                    kernel_col_idx = I.reshape(I.indices(kernel_cols), (1, 1, 3))
                    products = (
                        input[n, channel_idx, oh + kernel_row_idx, ow + kernel_col_idx]
                        * weight[oc, channel_idx, kernel_row_idx, kernel_col_idx]
                    )
                    value = I.reduce.sum(
                        products,
                        axis=(0, 1, 2),
                        acc_dtype=I.f32,
                    )
                    output[n, oc, oh, ow] = I.select(
                        value > 0.0,
                        value,
                        value * negative_slope,
                    )


def build(context):
    conv = context.compile("leaky_relu_conv2d", _conv2d_leaky, constexprs={})
    conv_no_bias = context.compile(
        "leaky_relu_conv2d_no_bias", _conv2d_leaky_no_bias, constexprs={}
    )

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
        if stride != 1 or padding != 0 or dilation != 1 or groups != 1:
            raise ValueError("only the fixed convolution profile is supported")
        if inplace:
            raise ValueError("inplace convolution output is not supported")

        output_shape = (1, 6, 766, 766)
        output = torch.empty(output_shape, device=input.device, dtype=input.dtype)
        if bias is None:
            conv_no_bias(input, weight, output, negative_slope)
        else:
            conv(input, weight, bias, output, negative_slope)
        return output

    return leaky_relu_conv2d
