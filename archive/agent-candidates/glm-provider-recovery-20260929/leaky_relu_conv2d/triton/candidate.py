from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _leaky_relu_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    input_stride_batch,
    input_stride_channel,
    input_stride_height,
    input_stride_width,
    output_stride_channel,
    output_stride_height,
    output_stride_width,
    negative_slope,
    in_channels_per_group: tl.constexpr,
    out_channels_per_group: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    has_bias: tl.constexpr,
    block_size: tl.constexpr,
):
    column_block = tl.program_id(0)
    row = tl.program_id(1)
    batch_group = tl.program_id(2)
    batch_id = batch_group // groups
    group_id = batch_group % groups

    columns = column_block * block_size + tl.arange(0, block_size)
    column_mask = columns < output_width
    input_group_offset = group_id * in_channels_per_group * input_stride_channel
    output_base = (
        output_ptr
        + batch_id * out_channels_per_group * groups * output_stride_channel
        + group_id * out_channels_per_group * output_stride_channel
    )
    input_base = input_ptr + batch_id * input_stride_batch + input_group_offset
    weight_base = weight_ptr + group_id * (
        out_channels_per_group * in_channels_per_group * kernel_height * kernel_width
    )

    accumulator_0 = tl.zeros([block_size], dtype=tl.float32)
    accumulator_1 = tl.zeros([block_size], dtype=tl.float32)
    accumulator_2 = tl.zeros([block_size], dtype=tl.float32)
    accumulator_3 = tl.zeros([block_size], dtype=tl.float32)
    accumulator_4 = tl.zeros([block_size], dtype=tl.float32)
    accumulator_5 = tl.zeros([block_size], dtype=tl.float32)

    for input_channel in tl.static_range(in_channels_per_group):
        channel_input = input_base + input_channel * input_stride_channel
        for kernel_row in tl.static_range(kernel_height):
            input_row = row * stride_height - padding_height + kernel_row * dilation_height
            row_valid = (input_row >= 0) & (input_row < input_height)
            row_input = channel_input + input_row * input_stride_height
            for kernel_column in tl.static_range(kernel_width):
                input_column = columns * stride_width - padding_width + kernel_column * dilation_width
                patch_mask = column_mask & row_valid & (input_column >= 0) & (input_column < input_width)
                patch = tl.load(
                    row_input + input_column * input_stride_width,
                    mask=patch_mask,
                    other=0.0,
                    cache_modifier=".ca",
                )
                weight_offset = (
                    input_channel * kernel_height + kernel_row
                ) * kernel_width + kernel_column
                weight_0 = tl.load(weight_base + weight_offset)
                weight_1 = tl.load(weight_base + out_channels_per_group * in_channels_per_group * kernel_height * kernel_width + weight_offset)
                weight_2 = tl.load(weight_base + 2 * out_channels_per_group * in_channels_per_group * kernel_height * kernel_width + weight_offset)
                weight_3 = tl.load(weight_base + 3 * out_channels_per_group * in_channels_per_group * kernel_height * kernel_width + weight_offset)
                weight_4 = tl.load(weight_base + 4 * out_channels_per_group * in_channels_per_group * kernel_height * kernel_width + weight_offset)
                weight_5 = tl.load(weight_base + 5 * out_channels_per_group * in_channels_per_group * kernel_height * kernel_width + weight_offset)
                accumulator_0 += patch * weight_0
                accumulator_1 += patch * weight_1
                accumulator_2 += patch * weight_2
                accumulator_3 += patch * weight_3
                accumulator_4 += patch * weight_4
                accumulator_5 += patch * weight_5

    if has_bias:
        bias_base = bias_ptr + group_id * out_channels_per_group
        accumulator_0 += tl.load(bias_base)
        accumulator_1 += tl.load(bias_base + 1)
        accumulator_2 += tl.load(bias_base + 2)
        accumulator_3 += tl.load(bias_base + 3)
        accumulator_4 += tl.load(bias_base + 4)
        accumulator_5 += tl.load(bias_base + 5)

    output_row = output_base + row * output_stride_height
    accumulator_0 = tl.where(accumulator_0 > 0.0, accumulator_0, accumulator_0 * negative_slope)
    accumulator_1 = tl.where(accumulator_1 > 0.0, accumulator_1, accumulator_1 * negative_slope)
    accumulator_2 = tl.where(accumulator_2 > 0.0, accumulator_2, accumulator_2 * negative_slope)
    accumulator_3 = tl.where(accumulator_3 > 0.0, accumulator_3, accumulator_3 * negative_slope)
    accumulator_4 = tl.where(accumulator_4 > 0.0, accumulator_4, accumulator_4 * negative_slope)
    accumulator_5 = tl.where(accumulator_5 > 0.0, accumulator_5, accumulator_5 * negative_slope)
    tl.store(
        output_row + columns * output_stride_width,
        accumulator_0,
        mask=column_mask,
    )
    tl.store(
        output_row + output_stride_channel + columns * output_stride_width,
        accumulator_1,
        mask=column_mask,
    )
    tl.store(
        output_row + 2 * output_stride_channel + columns * output_stride_width,
        accumulator_2,
        mask=column_mask,
    )
    tl.store(
        output_row + 3 * output_stride_channel + columns * output_stride_width,
        accumulator_3,
        mask=column_mask,
    )
    tl.store(
        output_row + 4 * output_stride_channel + columns * output_stride_width,
        accumulator_4,
        mask=column_mask,
    )
    tl.store(
        output_row + 5 * output_stride_channel + columns * output_stride_width,
        accumulator_5,
        mask=column_mask,
    )


def build(context):
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
        if input.ndim != 4 or weight.ndim != 4:
            raise ValueError("input and weight must be four-dimensional")
        if groups != 1 or weight.shape[0] != 6 or weight.shape[1] != input.shape[1]:
            raise ValueError("unsupported weight channel shape")

        stride_height, stride_width = (stride, stride) if isinstance(stride, int) else stride
        padding_height, padding_width = (padding, padding) if isinstance(padding, int) else padding
        dilation_height, dilation_width = (dilation, dilation) if isinstance(dilation, int) else dilation
        batch_size, in_channels, input_height, input_width = input.shape
        out_channels, _, kernel_height, kernel_width = weight.shape
        if out_channels % groups or in_channels % groups:
            raise ValueError("channels must be divisible by groups")
        out_channels_per_group = out_channels // groups
        in_channels_per_group = in_channels // groups
        output_height = (
            input_height + 2 * padding_height - dilation_height * (kernel_height - 1) - 1
        ) // stride_height + 1
        output_width = (
            input_width + 2 * padding_width - dilation_width * (kernel_width - 1) - 1
        ) // stride_width + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("convolution output is empty")
        if bias is not None and bias.shape != (out_channels,):
            raise ValueError("bias shape is incompatible with weight")

        output = torch.empty(
            (batch_size, out_channels, output_height, output_width),
            device=input.device,
            dtype=input.dtype,
        )

        _leaky_relu_conv2d_kernel[(triton.cdiv(output_width, 256), output_height, batch_size * groups)](
            input,
            weight,
            bias,
            output,
            input_height,
            input_width,
            output_height,
            output_width,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            negative_slope,
            in_channels_per_group=in_channels_per_group,
            out_channels_per_group=out_channels_per_group,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            stride_height=stride_height,
            stride_width=stride_width,
            padding_height=padding_height,
            padding_width=padding_width,
            dilation_height=dilation_height,
            dilation_width=dilation_width,
            groups=groups,
            has_bias=bias is not None,
            block_size=256,
            num_warps=8,
        )
        return output

    return leaky_relu_conv2d
