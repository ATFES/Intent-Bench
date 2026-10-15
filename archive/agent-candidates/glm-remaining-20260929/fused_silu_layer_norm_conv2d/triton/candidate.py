from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_stats_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    conv_ptr,
    sum_ptr,
    sqsum_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_channels,
    out_height,
    out_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    x_batch_stride,
    x_channel_stride,
    x_height_stride,
    x_width_stride,
    weight_out_stride,
    weight_in_stride,
    weight_height_stride,
    weight_width_stride,
    conv_batch_stride,
    conv_channel_stride,
    conv_height_stride,
    conv_width_stride,
    in_channels_per_group,
    out_channels_per_group,
    group_count,
    HAS_BIAS: tl.constexpr,
    KERNEL_HEIGHT: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bg = tl.program_id(2)
    batch_index = pid_bg // group_count
    group_index = pid_bg % group_count

    output_flat = out_height * out_width
    output_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    output_valid = output_offsets < output_flat
    output_y = output_offsets // out_width
    output_x = output_offsets % out_width

    channel_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    channel_valid = channel_offsets < out_channels_per_group
    output_channel = group_index * out_channels_per_group + channel_offsets
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kernel_y in tl.static_range(0, KERNEL_HEIGHT):
        input_y = output_y * stride_h - padding_h + kernel_y * dilation_h
        y_valid = (input_y >= 0) & (input_y < in_height)
        for kernel_x in tl.static_range(0, KERNEL_WIDTH):
            input_x = output_x * stride_w - padding_w + kernel_x * dilation_w
            x_valid = (input_x >= 0) & (input_x < in_width)
            for k_start in tl.range(0, in_channels_per_group, BLOCK_K):
                k_offsets = k_start + tl.arange(0, BLOCK_K)
                k_valid = k_offsets < in_channels_per_group
                input_channel = group_index * in_channels_per_group + k_offsets

                input_offsets = (
                    batch_index * x_batch_stride
                    + input_channel[:, None] * x_channel_stride
                    + input_y[None, :] * x_height_stride
                    + input_x[None, :] * x_width_stride
                )
                input_mask = (
                    k_valid[:, None]
                    & output_valid[None, :]
                    & y_valid[None, :]
                    & x_valid[None, :]
                )
                input_tile = tl.load(
                    x_ptr + input_offsets,
                    mask=input_mask,
                    other=0.0,
                )

                weight_offsets = (
                    output_channel[None, :] * weight_out_stride
                    + input_channel[:, None] * weight_in_stride
                    + kernel_y * weight_height_stride
                    + kernel_x * weight_width_stride
                )
                weight_mask = k_valid[:, None] & channel_valid[None, :]
                weight_tile = tl.load(
                    weight_ptr + weight_offsets,
                    mask=weight_mask,
                    other=0.0,
                )
                accumulator = tl.dot(
                    input_tile,
                    weight_tile,
                    accumulator,
                    input_precision="tf32x3",
                )

    if HAS_BIAS:
        bias_offsets = group_index * out_channels_per_group + channel_offsets
        accumulator += tl.load(
            bias_ptr + bias_offsets,
            mask=channel_valid,
            other=0.0,
        )[None, :]

    accumulator = tl.where(output_valid[:, None] & channel_valid[None, :], accumulator, 0.0)
    output_offsets_2d = (
        batch_index * conv_batch_stride
        + output_channel[None, :] * conv_channel_stride
        + output_y[:, None] * conv_height_stride
        + output_x[:, None] * conv_width_stride
    )
    tl.store(conv_ptr + output_offsets_2d, accumulator, mask=output_valid[:, None] & channel_valid[None, :])

    if pid_n == 0:
        tile_sum = tl.sum(tl.sum(accumulator, axis=1), axis=0)
        tile_sqsum = tl.sum(tl.sum(accumulator * accumulator, axis=1), axis=0)
        stat_index = batch_index * tl.num_programs(0) + pid_m
        tl.store(sum_ptr + stat_index, tile_sum)
        tl.store(sqsum_ptr + stat_index, tile_sqsum)


@triton.jit
def _normalize_silu_kernel(
    conv_ptr,
    sum_ptr,
    sqsum_ptr,
    output_ptr,
    element_count,
    stat_count,
    eps,
    conv_batch_stride,
    STAT_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    batch_index = tl.program_id(1)
    block_index = tl.program_id(0)
    offsets = block_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < element_count

    stat_sum = 0.0
    stat_sqsum = 0.0
    stat_offsets = tl.arange(0, STAT_BLOCK)
    for stat_start in tl.range(0, stat_count, STAT_BLOCK):
        stat_valid = stat_start + stat_offsets < stat_count
        stat_sum += tl.sum(
            tl.load(sum_ptr + stat_start + stat_offsets, mask=stat_valid, other=0.0),
            axis=0,
        )
        stat_sqsum += tl.sum(
            tl.load(sqsum_ptr + stat_start + stat_offsets, mask=stat_valid, other=0.0),
            axis=0,
        )

    mean = stat_sum / element_count
    variance = stat_sqsum / element_count - mean * mean
    variance = tl.maximum(variance, 0.0)
    inverse_std = tl.rsqrt(variance + eps)

    values = tl.load(conv_ptr + batch_index * conv_batch_stride + offsets, mask=valid, other=0.0)
    normalized = (values - mean) * inverse_std
    activated = normalized * tl.sigmoid(normalized)
    tl.store(output_ptr + batch_index * element_count + offsets, activated, mask=valid)


def build(context):
    def fused_silu_layer_norm_conv2d(
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor | None = None,
        conv_stride: int = 1,
        conv_padding: int = 0,
        conv_dilation: int = 1,
        conv_groups: int = 1,
        ln_eps: float = 1e-05,
    ) -> torch.Tensor:
        del weight

        batch_size, in_channels, in_height, in_width = x.shape
        out_channels, weight_in_channels, kernel_height, kernel_width = conv_weight.shape
        stride_h, stride_w = (conv_stride, conv_stride) if isinstance(conv_stride, int) else conv_stride
        padding_h, padding_w = (conv_padding, conv_padding) if isinstance(conv_padding, int) else conv_padding
        dilation_h, dilation_w = (conv_dilation, conv_dilation) if isinstance(conv_dilation, int) else conv_dilation
        groups = conv_groups

        out_height = (in_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1) // stride_h + 1
        out_width = (in_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1) // stride_w + 1
        output = torch.empty(
            (batch_size, out_channels, out_height, out_width),
            device=x.device,
            dtype=x.dtype,
        )
        if output.numel() == 0:
            return output

        in_channels_per_group = in_channels // groups
        out_channels_per_group = out_channels // groups
        output_flat = out_height * out_width
        blocks_per_batch = triton.cdiv(output_flat, 64)
        statistics = torch.empty(
            (2, batch_size * blocks_per_batch),
            device=x.device,
            dtype=torch.float32,
        )
        bias_pointer = conv_bias if conv_bias is not None else conv_weight

        _conv_stats_kernel[(blocks_per_batch, triton.cdiv(out_channels_per_group, 64), batch_size * groups)](
            x,
            conv_weight,
            bias_pointer,
            output,
            statistics[0],
            statistics[1],
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            out_height,
            out_width,
            kernel_height,
            kernel_width,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            conv_weight.stride(0),
            conv_weight.stride(1),
            conv_weight.stride(2),
            conv_weight.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            in_channels_per_group,
            out_channels_per_group,
            groups,
            HAS_BIAS=conv_bias is not None,
            KERNEL_HEIGHT=kernel_height,
            KERNEL_WIDTH=kernel_width,
            BLOCK_M=64,
            BLOCK_N=64,
            BLOCK_K=48,
        )
        _normalize_silu_kernel[(blocks_per_batch, batch_size)](
            output,
            statistics[0],
            statistics[1],
            output,
            output_flat,
            blocks_per_batch,
            ln_eps,
            output.stride(0),
            STAT_BLOCK=128,
            BLOCK_SIZE=4096,
        )
        return output

    return fused_silu_layer_norm_conv2d
