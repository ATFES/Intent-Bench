from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_selu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    groups,
    output_channels_per_group,
    kernel_size_hw,
    kernel_width,
    input_batch_stride,
    input_channel_stride,
    input_row_stride,
    input_col_stride,
    weight_out_stride,
    weight_channel_stride,
    weight_row_stride,
    weight_col_stride,
    output_batch_stride,
    output_channel_stride,
    output_row_stride,
    output_col_stride,
    group_input_channels,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    has_bias: tl.constexpr,
    tile_rows: tl.constexpr,
    tile_channels: tl.constexpr,
    reduction_tile: tl.constexpr,
):
    row_tile = tl.program_id(0)
    batch_group = tl.program_id(1)
    batch_index = batch_group // groups
    group_index = batch_group % groups
    output_channel_base = group_index * output_channels_per_group

    rows = row_tile * tile_rows + tl.arange(0, tile_rows)
    row_mask = rows < output_height * output_width
    output_col = rows % output_width
    output_row = rows // output_width
    center_y = output_row * stride_h - padding_h
    center_x = output_col * stride_w - padding_w

    output_channel = tl.arange(0, tile_channels)
    channel_mask = output_channel < output_channels_per_group
    result = tl.zeros((tile_rows, tile_channels), dtype=tl.float32)

    for reduction_start in tl.range(0, group_input_channels * kernel_size_hw, reduction_tile):
        reduction_index = reduction_start + tl.arange(0, reduction_tile)
        reduction_mask = reduction_index < group_input_channels * kernel_size_hw
        local_channel = reduction_index // kernel_size_hw
        kernel_offset = reduction_index % kernel_size_hw
        kernel_row = kernel_offset // kernel_width
        kernel_col = kernel_offset % kernel_width

        input_y = center_y[:, None] + kernel_row[None, :] * dilation_h
        input_x = center_x[:, None] + kernel_col[None, :] * dilation_w
        input_mask = (
            row_mask[:, None]
            & reduction_mask[None, :]
            & (input_y >= 0)
            & (input_y < input_height)
            & (input_x >= 0)
            & (input_x < input_width)
        )
        input_offset = (
            batch_index * input_batch_stride
            + local_channel[None, :] * input_channel_stride
            + input_y * input_row_stride
            + input_x * input_col_stride
        )
        input_values = tl.load(input_ptr + input_offset, mask=input_mask, other=0.0)

        weight_mask = reduction_mask[:, None] & channel_mask[None, :]
        weight_offset = (
            (output_channel_base + output_channel)[None, :] * weight_out_stride
            + reduction_index[:, None] * weight_channel_stride
            + kernel_row[:, None] * weight_row_stride
            + kernel_col[:, None] * weight_col_stride
        )
        weight_values = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0)
        result = tl.dot(
            input_values,
            tl.trans(weight_values),
            acc=result,
            input_precision="ieee",
        )

    if has_bias:
        bias = tl.load(bias_ptr + output_channel_base + output_channel, mask=channel_mask, other=0.0)
        result += bias[None, :]

    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    activated = scale * tl.where(
        result > 0.0,
        result,
        alpha * (tl.exp(tl.minimum(result, 0.0)) - 1.0),
    )

    output_offset = (
        batch_index * output_batch_stride
        + (output_channel_base + output_channel)[None, :] * output_channel_stride
        + output_row[:, None] * output_row_stride
        + output_col[:, None] * output_col_stride
    )
    tl.store(output_ptr + output_offset, activated, mask=row_mask[:, None] & channel_mask[None, :])


@triton.jit
def _instance_norm_kernel(
    input_ptr,
    output_ptr,
    count,
    input_batch_stride,
    input_channel_stride,
    input_element_stride,
    output_batch_stride,
    output_channel_stride,
    output_element_stride,
    eps,
    block_size: tl.constexpr,
    single_tile: tl.constexpr,
):
    channel = tl.program_id(0)
    batch_index = tl.program_id(1)
    input_base = input_ptr + batch_index * input_batch_stride + channel * input_channel_stride
    output_base = output_ptr + batch_index * output_batch_stride + channel * output_channel_stride

    if single_tile:
        offsets = tl.arange(0, block_size)
        mask = offsets < count
        values = tl.load(input_base + offsets * input_element_stride, mask=mask, other=0.0)
        mean = tl.sum(values, axis=0) / count
        centered_square_sum = tl.sum((values - mean) * (values - mean), axis=0)
        variance = centered_square_sum / count
        reciprocal_std = 1.0 / tl.sqrt_rn(variance + eps)
        normalized = (values - mean) * reciprocal_std
        tl.store(output_base + offsets * output_element_stride, normalized, mask=mask)
    else:
        values_sum = 0.0
        square_sum = 0.0
        for start in tl.range(0, count, block_size):
            offsets = start + tl.arange(0, block_size)
            mask = offsets < count
            values = tl.load(input_base + offsets * input_element_stride, mask=mask, other=0.0)
            values_sum += tl.sum(values, axis=0)
            square_sum += tl.sum(values * values, axis=0)
        mean = values_sum / count
        variance = square_sum / count - mean * mean
        variance = tl.maximum(variance, 0.0)
        reciprocal_std = 1.0 / tl.sqrt_rn(variance + eps)
        for start in tl.range(0, count, block_size):
            offsets = start + tl.arange(0, block_size)
            mask = offsets < count
            values = tl.load(input_base + offsets * input_element_stride, mask=mask, other=0.0)
            normalized = (values - mean) * reciprocal_std
            tl.store(output_base + offsets * output_element_stride, normalized, mask=mask)


def _pair(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("stride, padding, and dilation must contain two values")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def build(context):
    def fused_instance_norm_selu_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        num_features=None,
        eps=1e-05,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
    ) -> torch.Tensor:
        if input.ndim != 4 or weight.ndim != 4:
            raise ValueError("input and weight must be four-dimensional")
        if input.shape[1] % groups != 0 or weight.shape[0] % groups != 0:
            raise ValueError("channels must be divisible by groups")
        if weight.shape[1] != input.shape[1] // groups:
            raise ValueError("weight input channels do not match input")

        conv_stride = _pair(stride)
        conv_padding = _pair(padding)
        conv_dilation = _pair(dilation)
        batch, in_channels, input_height, input_width = input.shape
        out_channels, group_input_channels, kernel_height, kernel_width = weight.shape
        output_height = (
            input_height + 2 * conv_padding[0] - conv_dilation[0] * (kernel_height - 1) - 1
        ) // conv_stride[0] + 1
        output_width = (
            input_width + 2 * conv_padding[1] - conv_dilation[1] * (kernel_width - 1) - 1
        ) // conv_stride[1] + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("convolution output is empty")
        if num_features is not None and int(num_features) != out_channels:
            raise ValueError("num_features does not match convolution output channels")

        output = torch.empty(
            (batch, out_channels, output_height, output_width),
            device=input.device,
            dtype=input.dtype,
        )
        output_channels_per_group = out_channels // groups
        patch_count = output_height * output_width
        tile_rows = 64
        tile_channels = triton.next_power_of_2(output_channels_per_group)

        _conv_selu_kernel[(triton.cdiv(patch_count, tile_rows), batch * groups)](
            input,
            weight,
            bias if bias is not None else weight,
            output,
            batch,
            input_height,
            input_width,
            out_channels,
            output_height,
            output_width,
            groups,
            output_channels_per_group,
            kernel_height * kernel_width,
            kernel_width,
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
            output.stride(2),
            output.stride(3),
            group_input_channels,
            conv_stride[0],
            conv_stride[1],
            conv_padding[0],
            conv_padding[1],
            conv_dilation[0],
            conv_dilation[1],
            has_bias=bias is not None,
            tile_rows=tile_rows,
            tile_channels=tile_channels,
            reduction_tile=16,
            num_warps=4,
        )

        spatial_count = output_height * output_width
        block_size = min(triton.next_power_of_2(spatial_count), 4096)
        _instance_norm_kernel[(out_channels, batch)](
            output,
            output,
            spatial_count,
            output.stride(0),
            output.stride(1),
            output.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(3),
            eps,
            block_size=block_size,
            single_tile=spatial_count <= block_size,
            num_warps=8,
        )
        return output

    return fused_instance_norm_selu_conv2d
