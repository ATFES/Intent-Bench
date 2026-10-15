import torch
import intent
import intent.language as I


@intent.kernel
def _sigmoid_conv2d_bias(
    input: I.In[I.f32, (1, 64, 64, 64)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    bias: I.In[I.f32, (128,)],
    out: I.Out[I.f32, (1, 128, 62, 62)],
):
    batch = I.domain(0, 1)
    output_channel = I.domain(0, 128)
    output_row = I.domain(0, 62)
    output_col = I.domain(0, 62)
    input_channel = I.domain(0, 64)
    kernel_row = I.domain(0, 3)
    kernel_col = I.domain(0, 3)

    batch_index = I.reshape(I.indices(batch), (1, 1, 1, 1, 1, 1, 1))
    channel_index = I.reshape(
        I.indices(output_channel), (1, 128, 1, 1, 1, 1, 1)
    )
    row_index = I.reshape(I.indices(output_row), (1, 1, 62, 1, 1, 1, 1))
    col_index = I.reshape(I.indices(output_col), (1, 1, 1, 62, 1, 1, 1))
    input_channel_index = I.reshape(
        I.indices(input_channel), (1, 1, 1, 1, 64, 1, 1)
    )
    kernel_row_index = I.reshape(
        I.indices(kernel_row), (1, 1, 1, 1, 1, 3, 1)
    )
    kernel_col_index = I.reshape(
        I.indices(kernel_col), (1, 1, 1, 1, 1, 1, 3)
    )

    input_value = input[
        batch_index,
        input_channel_index,
        row_index + kernel_row_index,
        col_index + kernel_col_index,
    ]
    weight_value = weight[
        channel_index,
        input_channel_index,
        kernel_row_index,
        kernel_col_index,
    ]
    convolution = I.reduce.sum(
        input_value * weight_value,
        axis=(4, 5, 6),
        acc_dtype=I.f32,
    )
    biased = convolution + I.reshape(bias[output_channel], (1, 128, 1, 1))
    out[batch, output_channel, output_row, output_col] = I.sigmoid(biased)


@intent.kernel
def _sigmoid_conv2d_no_bias(
    input: I.In[I.f32, (1, 64, 64, 64)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    out: I.Out[I.f32, (1, 128, 62, 62)],
):
    batch = I.domain(0, 1)
    output_channel = I.domain(0, 128)
    output_row = I.domain(0, 62)
    output_col = I.domain(0, 62)
    input_channel = I.domain(0, 64)
    kernel_row = I.domain(0, 3)
    kernel_col = I.domain(0, 3)

    batch_index = I.reshape(I.indices(batch), (1, 1, 1, 1, 1, 1, 1))
    channel_index = I.reshape(
        I.indices(output_channel), (1, 128, 1, 1, 1, 1, 1)
    )
    row_index = I.reshape(I.indices(output_row), (1, 1, 62, 1, 1, 1, 1))
    col_index = I.reshape(I.indices(output_col), (1, 1, 1, 62, 1, 1, 1))
    input_channel_index = I.reshape(
        I.indices(input_channel), (1, 1, 1, 1, 64, 1, 1)
    )
    kernel_row_index = I.reshape(
        I.indices(kernel_row), (1, 1, 1, 1, 1, 3, 1)
    )
    kernel_col_index = I.reshape(
        I.indices(kernel_col), (1, 1, 1, 1, 1, 1, 3)
    )

    input_value = input[
        batch_index,
        input_channel_index,
        row_index + kernel_row_index,
        col_index + kernel_col_index,
    ]
    weight_value = weight[
        channel_index,
        input_channel_index,
        kernel_row_index,
        kernel_col_index,
    ]
    convolution = I.reduce.sum(
        input_value * weight_value,
        axis=(4, 5, 6),
        acc_dtype=I.f32,
    )
    out[batch, output_channel, output_row, output_col] = I.sigmoid(convolution)


def build(context):
    bias_kernel = context.compile(
        "sigmoid_conv2d_bias", _sigmoid_conv2d_bias, constexprs={}
    )
    no_bias_kernel = context.compile(
        "sigmoid_conv2d_no_bias", _sigmoid_conv2d_no_bias, constexprs={}
    )

    def sigmoid_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        out=None,
    ):
        if out is None:
            out = torch.empty(
                (1, 128, 62, 62), device=input.device, dtype=input.dtype
            )
        if bias is None:
            no_bias_kernel(input, weight, out)
        else:
            bias_kernel(input, weight, bias, out)
        return out

    return sigmoid_conv2d
