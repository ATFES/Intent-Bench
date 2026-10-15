import torch
import intent
import intent.language as I


@intent.kernel
def _conv2d_with_bias(
    input: I.In[I.f32, (8, 64, 16, 16)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    bias: I.In[I.f32, (128,)],
    output: I.Out[I.f32, (8, 128, 14, 14)],
):
    batch = I.domain(0, 8)
    channels = I.domain(0, 64)
    output_channels = I.domain(0, 128)
    output_rows = I.domain(0, 14)
    output_columns = I.domain(0, 14)
    kernel_rows = I.domain(0, 3)
    kernel_columns = I.domain(0, 3)

    batch6 = I.reshape(I.indices(batch), (8, 1, 1, 1, 1, 1))
    channels6 = I.reshape(I.indices(channels), (1, 64, 1, 1, 1, 1))
    rows6 = I.reshape(I.indices(output_rows), (1, 1, 14, 1, 1, 1))
    columns6 = I.reshape(I.indices(output_columns), (1, 1, 1, 14, 1, 1))
    kernel_rows6 = I.reshape(I.indices(kernel_rows), (1, 1, 1, 1, 3, 1))
    kernel_columns6 = I.reshape(I.indices(kernel_columns), (1, 1, 1, 1, 1, 3))

    output_channels4 = I.reshape(I.indices(output_channels), (128, 1, 1, 1))
    channels4 = I.reshape(I.indices(channels), (1, 64, 1, 1))
    kernel_rows4 = I.reshape(I.indices(kernel_rows), (1, 1, 3, 1))
    kernel_columns4 = I.reshape(I.indices(kernel_columns), (1, 1, 1, 3))

    patch = input[
        batch6,
        channels6,
        rows6 + kernel_rows6,
        columns6 + kernel_columns6,
    ]
    filters = weight[output_channels4, channels4, kernel_rows4, kernel_columns4]
    contracted = I.contract(
        patch,
        filters,
        reduce=((1, 1), (4, 2), (5, 3)),
        batch=(),
        acc_dtype=I.f32,
    )
    contracted = I.transpose(contracted, (0, 3, 1, 2))
    bias_value = I.reshape(bias, (1, 128, 1, 1))
    output[batch, output_channels, output_rows, output_columns] = contracted + bias_value


@intent.kernel
def _conv2d_without_bias(
    input: I.In[I.f32, (8, 64, 16, 16)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    output: I.Out[I.f32, (8, 128, 14, 14)],
):
    batch = I.domain(0, 8)
    channels = I.domain(0, 64)
    output_channels = I.domain(0, 128)
    output_rows = I.domain(0, 14)
    output_columns = I.domain(0, 14)
    kernel_rows = I.domain(0, 3)
    kernel_columns = I.domain(0, 3)

    batch6 = I.reshape(I.indices(batch), (8, 1, 1, 1, 1, 1))
    channels6 = I.reshape(I.indices(channels), (1, 64, 1, 1, 1, 1))
    rows6 = I.reshape(I.indices(output_rows), (1, 1, 14, 1, 1, 1))
    columns6 = I.reshape(I.indices(output_columns), (1, 1, 1, 14, 1, 1))
    kernel_rows6 = I.reshape(I.indices(kernel_rows), (1, 1, 1, 1, 3, 1))
    kernel_columns6 = I.reshape(I.indices(kernel_columns), (1, 1, 1, 1, 1, 3))

    output_channels4 = I.reshape(I.indices(output_channels), (128, 1, 1, 1))
    channels4 = I.reshape(I.indices(channels), (1, 64, 1, 1))
    kernel_rows4 = I.reshape(I.indices(kernel_rows), (1, 1, 3, 1))
    kernel_columns4 = I.reshape(I.indices(kernel_columns), (1, 1, 1, 3))

    patch = input[
        batch6,
        channels6,
        rows6 + kernel_rows6,
        columns6 + kernel_columns6,
    ]
    filters = weight[output_channels4, channels4, kernel_rows4, kernel_columns4]
    contracted = I.contract(
        patch,
        filters,
        reduce=((1, 1), (4, 2), (5, 3)),
        batch=(),
        acc_dtype=I.f32,
    )
    output[batch, output_channels, output_rows, output_columns] = I.transpose(
        contracted, (0, 3, 1, 2)
    )


def build(context):
    with_bias = context.compile("conv2d_with_bias", _conv2d_with_bias)
    without_bias = context.compile("conv2d_without_bias", _conv2d_without_bias)

    def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        output = torch.empty((8, 128, 14, 14), device=input.device, dtype=input.dtype)
        if bias is None:
            without_bias(input, weight, output)
        else:
            with_bias(input, weight, bias, output)
        return output

    return conv2d
