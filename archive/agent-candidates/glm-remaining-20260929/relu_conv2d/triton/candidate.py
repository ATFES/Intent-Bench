from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _relu_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    weight_output_stride,
    weight_input_stride,
    weight_height_stride,
    weight_width_stride,
    output_batch_stride,
    output_channel_stride,
    bias_stride,
    input_height,
    input_width,
    output_height,
    output_width,
    output_channels_per_group,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    input_channels_per_group: tl.constexpr,
    kernel_height: tl.constexpr,
    kernel_width: tl.constexpr,
    output_channels: tl.constexpr,
    block_hw: tl.constexpr,
    block_oc: tl.constexpr,
    has_bias: tl.constexpr,
    use_dot: tl.constexpr,
):
    reduction_size: tl.constexpr = input_channels_per_group * kernel_height * kernel_width
    hw_tile_count = tl.num_programs(0) // output_channels
    pid_hw = tl.program_id(0) % hw_tile_count
    pid_oc = tl.program_id(0) // hw_tile_count
    pid_batch = tl.program_id(1)
    pid_group = tl.program_id(2)

    output_hw = output_height * output_width
    output_offsets = pid_hw * block_hw + tl.arange(0, block_hw)
    output_offsets = output_offsets.to(tl.int64)
    output_row = output_offsets // output_width
    output_col = output_offsets % output_width
    hw_valid = output_offsets < output_hw

    output_channel_offsets = pid_oc * block_oc + tl.arange(0, block_oc)
    output_channel_offsets = output_channel_offsets.to(tl.int64)
    channel_valid = output_channel_offsets < output_channels_per_group

    input_batch_offset = pid_batch.to(tl.int64) * input_batch_stride
    output_batch_offset = pid_batch.to(tl.int64) * output_batch_stride
    global_output_channels = pid_group.to(tl.int64) * output_channels_per_group + output_channel_offsets
    output_channel_base = global_output_channels * output_channel_stride
    weight_output_base = global_output_channels * weight_output_stride

    accumulator = tl.zeros((block_hw, block_oc), dtype=tl.float32)
    reduction_offsets = tl.arange(0, reduction_size)

    if use_dot:
        output_rows = output_row[:, None]
        output_cols = output_col[:, None]
        kernel_extent = kernel_height * kernel_width
        input_channels = reduction_offsets[None, :] // kernel_extent
        kernel_remainder = reduction_offsets[None, :] % kernel_extent
        input_rows = (
            output_rows * stride_h - padding_h
            + (kernel_remainder // kernel_width) * dilation_h
        )
        input_cols = output_cols * stride_w - padding_w + (kernel_remainder % kernel_width) * dilation_w

        input_offsets = (
            input_batch_offset
            + input_channels.to(tl.int64) * input_channel_stride
            + input_rows.to(tl.int64) * input_height_stride
            + input_cols.to(tl.int64) * input_width_stride
        )
        patch_valid = hw_valid[:, None] & (input_rows >= 0) & (input_rows < input_height) & (input_cols >= 0) & (input_cols < input_width)
        patches = tl.load(input_ptr + input_offsets, mask=patch_valid, other=0.0)

        weight_channels = reduction_offsets[:, None] // kernel_extent
        weight_remainder = reduction_offsets[:, None] % kernel_extent
        weight_rows = weight_remainder // kernel_width
        weight_cols = weight_remainder % kernel_width
        weight_offsets = (
            weight_output_base[None, :]
            + weight_channels.to(tl.int64) * weight_input_stride
            + weight_rows.to(tl.int64) * weight_height_stride
            + weight_cols.to(tl.int64) * weight_width_stride
        )
        weights = tl.load(weight_ptr + weight_offsets, mask=channel_valid[None, :], other=0.0)
        accumulator = tl.dot(patches, weights, accumulator, input_precision="ieee")
    else:
        for reduction_index in tl.static_range(reduction_size):
            kernel_extent = kernel_height * kernel_width
            input_channel = reduction_index // kernel_extent
            kernel_remainder = reduction_index % kernel_extent
            kernel_row = kernel_remainder // kernel_width
            kernel_col = kernel_remainder % kernel_width
            input_rows = output_row * stride_h - padding_h + kernel_row * dilation_h
            input_cols = output_col * stride_w - padding_w + kernel_col * dilation_w
            input_offsets = (
                input_batch_offset
                + input_channel * input_channel_stride
                + input_rows.to(tl.int64) * input_height_stride
                + input_cols.to(tl.int64) * input_width_stride
            )
            input_valid = hw_valid & (input_rows >= 0) & (input_rows < input_height) & (input_cols >= 0) & (input_cols < input_width)
            inputs = tl.load(input_ptr + input_offsets, mask=input_valid, other=0.0)

            weight_offsets = (
                weight_output_base
                + input_channel * weight_input_stride
                + kernel_row * weight_height_stride
                + kernel_col * weight_width_stride
            )
            weights = tl.load(weight_ptr + weight_offsets, mask=channel_valid, other=0.0)
            accumulator += inputs[:, None] * weights[None, :]

    if has_bias:
        bias_offsets = global_output_channels * bias_stride
        bias_values = tl.load(bias_ptr + bias_offsets, mask=channel_valid, other=0.0)
        accumulator += bias_values[None, :]

    accumulator = tl.maximum(accumulator, 0.0, propagate_nan=tl.PropagateNan.ALL)
    output_offsets_2d = output_batch_offset + output_channel_base[None, :] + output_offsets[:, None]
    output_valid = hw_valid[:, None] & channel_valid[None, :]
    tl.store(output_ptr + output_offsets_2d, accumulator.to(output_ptr.dtype.element_ty), mask=output_valid)


def _pair(value, name):
    if isinstance(value, str):
        if value == "same":
            return ("same", "same"), (1, 1)
        if value == "valid":
            return (0, 0), (1, 1)
        raise ValueError(f"invalid padding string: {value!r}")
    if isinstance(value, int):
        return (value, value), (1, 1)
    if len(value) != 2:
        raise ValueError(f"invalid {name}: {value!r}")
    return (int(value[0]), int(value[1])), (1, 1)


def build(context):
    def relu_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        inplace=False,
    ):
        if input.ndim != 4 or weight.ndim != 4:
            raise ValueError("input and weight must be 4-dimensional")
        if groups <= 0 or input.shape[1] % groups or weight.shape[0] % groups:
            raise ValueError("incompatible groups and channel counts")
        if weight.shape[1] != input.shape[1] // groups:
            raise ValueError("weight input-channel count does not match input")
        if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
            raise ValueError("bias must be one-dimensional with weight.shape[0] elements")

        stride_pair, _ = _pair(stride, "stride")
        padding_pair, dilation_multiplier = _pair(padding, "padding")
        dilation_pair, _ = _pair(dilation, "dilation")
        if any(value < 0 for value in padding_pair):
            raise ValueError("padding must be non-negative")
        if any(value <= 0 for value in stride_pair + dilation_pair):
            raise ValueError("stride and dilation must be positive")
        input_height, input_width = int(input.shape[2]), int(input.shape[3])
        kernel_height, kernel_width = int(weight.shape[2]), int(weight.shape[3])
        if padding_pair == ("same", "same"):
            padding_pair = (
                dilation_pair[0] * (kernel_height - 1) // 2,
                dilation_pair[1] * (kernel_width - 1) // 2,
            )
        dilation_pair = tuple(value * dilation_multiplier for value in dilation_pair)
        kernel_extent_h = dilation_pair[0] * (kernel_height - 1) + 1
        kernel_extent_w = dilation_pair[1] * (kernel_width - 1) + 1
        output_height = (input_height + 2 * padding_pair[0] - kernel_extent_h) // stride_pair[0] + 1
        output_width = (input_width + 2 * padding_pair[1] - kernel_extent_w) // stride_pair[1] + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("calculated output shape is empty")

        output = torch.empty(
            (input.shape[0], weight.shape[0], output_height, output_width),
            dtype=input.dtype,
            device=input.device,
        )
        batch = int(input.shape[0])
        output_hw = output_height * output_width
        output_channels_per_group = int(weight.shape[0] // groups)
        input_channels_per_group = int(weight.shape[1])
        reduction_size = input_channels_per_group * kernel_height * kernel_width
        block_hw = 128 if output_hw >= 128 else 16
        block_oc = min(64, output_channels_per_group)
        if block_oc & (block_oc - 1):
            block_oc = 1
        use_dot = reduction_size >= 16 and block_hw >= 16 and block_oc >= 16

        hw_tiles = triton.cdiv(output_hw, block_hw)
        oc_tiles = triton.cdiv(output_channels_per_group, block_oc)
        grid = (hw_tiles * oc_tiles, batch, groups)
        _relu_conv2d_kernel[grid](
            input,
            weight,
            bias if bias is not None else input,
            output,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            weight.stride(0),
            weight.stride(1),
            weight.stride(2),
            weight.stride(3),
            output.stride(0),
            output.stride(1),
            bias.stride(0) if bias is not None else 0,
            input_height,
            input_width,
            output_height,
            output_width,
            output_channels_per_group,
            stride_pair[0],
            stride_pair[1],
            padding_pair[0],
            padding_pair[1],
            dilation_pair[0],
            dilation_pair[1],
            input_channels_per_group=input_channels_per_group,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            output_channels=output_channels_per_group,
            block_hw=block_hw,
            block_oc=block_oc,
            has_bias=bias is not None,
            use_dot=use_dot,
            num_warps=8,
        )
        return output

    return relu_conv2d
