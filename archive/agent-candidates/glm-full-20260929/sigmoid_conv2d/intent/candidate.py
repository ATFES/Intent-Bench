from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _sigmoid_conv2d_with_bias(
    input: I.In[I.f32, ("BATCH", "INPUT_CHANNELS", "INPUT_HEIGHT", "INPUT_WIDTH")],
    weight: I.In[I.f32, ("OUTPUT_CHANNELS", "INPUT_CHANNELS", "KERNEL_HEIGHT", "KERNEL_WIDTH")],
    bias: I.In[I.f32, ("OUTPUT_CHANNELS",)],
    output: I.Out[I.f32, ("BATCH", "OUTPUT_CHANNELS", "OUTPUT_HEIGHT", "OUTPUT_WIDTH")],
    stride_height: I.i32,
    stride_width: I.i32,
    padding_height: I.i32,
    padding_width: I.i32,
    dilation_height: I.i32,
    dilation_width: I.i32,
):
    input_channels = input.shape[1]
    output_height = output.shape[2]
    output_width = output.shape[3]
    kernel_height = weight.shape[2]
    kernel_width = weight.shape[3]

    output_points = I.domain(0, output_height * output_width)
    reduction_points = I.domain(
        0, input_channels * kernel_height * kernel_width
    )
    output_columns = I.domain(0, output.shape[1])

    row_coordinates = I.reshape(I.indices(output_points), (output_height * output_width, 1))
    reduction_coordinates = I.reshape(
        I.indices(reduction_points), (1, input_channels * kernel_height * kernel_width)
    )
    output_height_coordinate = row_coordinates // output_width
    output_width_coordinate = row_coordinates % output_width
    kernel_area = kernel_height * kernel_width
    channel_coordinate = reduction_coordinates // kernel_area
    kernel_coordinate = reduction_coordinates % kernel_area
    horizontal_kernel = (kernel_coordinate // kernel_width) * I.cast(
        dilation_width, I.index
    )
    vertical_kernel = (kernel_coordinate % kernel_width) * I.cast(
        dilation_height, I.index
    )

    input_height_coordinates = (
        output_height_coordinate * I.cast(stride_height, I.index)
        + vertical_kernel
        - I.cast(padding_height, I.index)
    )
    input_width_coordinates = (
        output_width_coordinate * I.cast(stride_width, I.index)
        + horizontal_kernel
        - I.cast(padding_width, I.index)
    )

    I.assume_in_bounds(channel_coordinate, input, 1)
    I.assume_in_bounds(input_height_coordinates, input, 2)
    I.assume_in_bounds(input_width_coordinates, input, 3)
    patches = input[
        0, channel_coordinate, input_height_coordinates, input_width_coordinates
    ]

    reduction_size = input_channels * kernel_area
    filters = I.transpose(
        I.reshape(weight, (output.shape[1], reduction_size)), (1, 0)
    )
    convolution = I.matmul(patches, filters, acc_dtype=I.f32)
    activated = I.sigmoid(convolution + bias)
    output[output_points, output_columns] = activated


@intent.kernel
def _sigmoid_conv2d_without_bias(
    input: I.In[I.f32, ("BATCH", "INPUT_CHANNELS", "INPUT_HEIGHT", "INPUT_WIDTH")],
    weight: I.In[I.f32, ("OUTPUT_CHANNELS", "INPUT_CHANNELS", "KERNEL_HEIGHT", "KERNEL_WIDTH")],
    output: I.Out[I.f32, ("BATCH", "OUTPUT_CHANNELS", "OUTPUT_HEIGHT", "OUTPUT_WIDTH")],
    stride_height: I.i32,
    stride_width: I.i32,
    padding_height: I.i32,
    padding_width: I.i32,
    dilation_height: I.i32,
    dilation_width: I.i32,
):
    input_channels = input.shape[1]
    output_height = output.shape[2]
    output_width = output.shape[3]
    kernel_height = weight.shape[2]
    kernel_width = weight.shape[3]

    output_points = I.domain(0, output_height * output_width)
    reduction_points = I.domain(
        0, input_channels * kernel_height * kernel_width
    )
    output_columns = I.domain(0, output.shape[1])

    row_coordinates = I.reshape(I.indices(output_points), (output_height * output_width, 1))
    reduction_coordinates = I.reshape(
        I.indices(reduction_points), (1, input_channels * kernel_height * kernel_width)
    )
    output_height_coordinate = row_coordinates // output_width
    output_width_coordinate = row_coordinates % output_width
    kernel_area = kernel_height * kernel_width
    channel_coordinate = reduction_coordinates // kernel_area
    kernel_coordinate = reduction_coordinates % kernel_area
    horizontal_kernel = (kernel_coordinate // kernel_width) * I.cast(
        dilation_width, I.index
    )
    vertical_kernel = (kernel_coordinate % kernel_width) * I.cast(
        dilation_height, I.index
    )

    input_height_coordinates = (
        output_height_coordinate * I.cast(stride_height, I.index)
        + vertical_kernel
        - I.cast(padding_height, I.index)
    )
    input_width_coordinates = (
        output_width_coordinate * I.cast(stride_width, I.index)
        + horizontal_kernel
        - I.cast(padding_width, I.index)
    )

    I.assume_in_bounds(channel_coordinate, input, 1)
    I.assume_in_bounds(input_height_coordinates, input, 2)
    I.assume_in_bounds(input_width_coordinates, input, 3)
    patches = input[
        0, channel_coordinate, input_height_coordinates, input_width_coordinates
    ]

    reduction_size = input_channels * kernel_area
    filters = I.transpose(
        I.reshape(weight, (output.shape[1], reduction_size)), (1, 0)
    )
    convolution = I.matmul(patches, filters, acc_dtype=I.f32)
    output[output_points, output_columns] = I.sigmoid(convolution)


def _pair(value, name):
    if isinstance(value, str):
        if value == "valid":
            return (0, 0)
        if value == "same":
            return None
        raise ValueError(f"unsupported {name} string: {value!r}")
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    return (int(value[0]), int(value[1]))


def build(context):
    with_bias = context.compile(
        "sigmoid_conv2d_implicit_gemm_bias",
        _sigmoid_conv2d_with_bias,
    )
    without_bias = context.compile(
        "sigmoid_conv2d_implicit_gemm_no_bias",
        _sigmoid_conv2d_without_bias,
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
        if input.ndim != 4 or weight.ndim != 4:
            raise ValueError("input and weight must be four-dimensional")
        if groups != 1:
            raise ValueError("this specialization supports groups=1")
        if input.shape[1] != weight.shape[1]:
            raise ValueError("input and weight channels do not match")
        if input.shape[0] != 1:
            raise ValueError("this specialization supports batch size 1")

        stride_height, stride_width = _pair(stride, "stride")
        dilation_height, dilation_width = _pair(dilation, "dilation")
        padding_values = _pair(padding, "padding")
        if padding_values is None:
            if stride_height != 1 or stride_width != 1:
                raise ValueError("padding='same' requires stride=1")
            padding_height = dilation_height * (weight.shape[2] - 1) // 2
            padding_width = dilation_width * (weight.shape[3] - 1) // 2
        else:
            padding_height, padding_width = padding_values

        input_height = input.shape[2]
        input_width = input.shape[3]
        denominator_height = dilation_height * (weight.shape[2] - 1) + 1
        denominator_width = dilation_width * (weight.shape[3] - 1) + 1
        output_height = (
            input_height + 2 * padding_height - denominator_height
        ) // stride_height + 1
        output_width = (
            input_width + 2 * padding_width - denominator_width
        ) // stride_width + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("convolution output is empty")

        output_shape = (1, weight.shape[0], output_height, output_width)
        if out is None:
            output = torch.empty(
                output_shape, dtype=input.dtype, device=input.device
            )
        else:
            if tuple(out.shape) != output_shape:
                raise ValueError("out has an incompatible shape")
            if out.dtype != input.dtype or out.device != input.device:
                raise ValueError("out has an incompatible dtype or device")
            output = out

        arguments = (
            input,
            weight,
            stride_height,
            stride_width,
            padding_height,
            padding_width,
            dilation_height,
            dilation_width,
            output,
        )
        if bias is None:
            without_bias(*arguments)
        else:
            if tuple(bias.shape) != (weight.shape[0],):
                raise ValueError("bias has an incompatible shape")
            with_bias(
                input,
                weight,
                bias,
                stride_height,
                stride_width,
                padding_height,
                padding_width,
                dilation_height,
                dilation_width,
                output,
            )
        return output

    return sigmoid_conv2d
