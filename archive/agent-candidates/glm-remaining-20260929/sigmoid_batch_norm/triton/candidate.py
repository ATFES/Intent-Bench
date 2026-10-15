from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _sigmoid_batch_norm_2d_kernel(
    input_ptr,
    output_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    num_channels,
    input_row_stride,
    output_row_stride,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    channel_start = tl.program_id(0) * BLOCK_CHANNELS
    row_start = tl.program_id(1) * BLOCK_ROWS

    channels = channel_start + tl.arange(0, BLOCK_CHANNELS)
    channel_mask = channels < num_channels
    mean = tl.load(mean_ptr + channels, mask=channel_mask, other=0.0)
    variance = tl.load(var_ptr + channels, mask=channel_mask, other=0.0)
    inv_std = tl.rsqrt(variance + eps)

    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + channels, mask=channel_mask, other=0.0)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_ptr + channels, mask=channel_mask, other=0.0)
    else:
        bias = 0.0

    rows = row_start + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < tl.num_programs(1) * BLOCK_ROWS
    input_rows = input_ptr + rows[:, None] * input_row_stride + channels[None, :]
    output_rows = output_ptr + rows[:, None] * output_row_stride + channels[None, :]
    mask = row_mask[:, None] & channel_mask[None, :]
    values = tl.load(input_rows, mask=mask, other=0.0)
    normalized = (values - mean) * inv_std * weight + bias
    result = tl.sigmoid(normalized)
    tl.store(output_rows, result, mask=mask)


@triton.jit
def _sigmoid_batch_norm_eval_generic_kernel(
    input_ptr,
    output_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    batch_count,
    sequence_length,
    input_batch_stride,
    input_channel_stride,
    input_position_stride,
    output_batch_stride,
    output_channel_stride,
    output_position_stride,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel = tl.program_id(0)
    block_start = tl.program_id(1) * BLOCK_SIZE
    positions = block_start + tl.arange(0, BLOCK_SIZE)
    mask = positions < batch_count
    sequence_index = positions // sequence_length
    sequence_position = positions % sequence_length

    mean = tl.load(mean_ptr + channel)
    variance = tl.load(var_ptr + channel)
    inv_std = tl.rsqrt(variance + eps)
    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + channel)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_ptr + channel)
    else:
        bias = 0.0

    input_offsets = (
        sequence_index * input_batch_stride
        + channel * input_channel_stride
        + sequence_position * input_position_stride
    )
    output_offsets = (
        sequence_index * output_batch_stride
        + channel * output_channel_stride
        + sequence_position * output_position_stride
    )
    values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    normalized = (values - mean) * inv_std * weight + bias
    tl.store(output_ptr + output_offsets, tl.sigmoid(normalized), mask=mask)


@triton.jit
def _sigmoid_batch_norm_training_kernel(
    input_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    batch_count,
    sequence_length,
    input_batch_stride,
    input_channel_stride,
    input_position_stride,
    output_batch_stride,
    output_channel_stride,
    output_position_stride,
    momentum,
    eps,
    HAS_WEIGHT: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    channel = tl.program_id(0)
    count = batch_count * sequence_length
    welford_mean = 0.0
    m2 = 0.0
    seen_count = 0.0
    for block_start in range(0, count, BLOCK_SIZE):
        positions = block_start + tl.arange(0, BLOCK_SIZE)
        mask = positions < count
        sequence_index = positions // sequence_length
        sequence_position = positions % sequence_length
        offsets = (
            sequence_index * input_batch_stride
            + channel * input_channel_stride
            + sequence_position * input_position_stride
        )
        values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        visible_count = tl.sum(mask.to(tl.float32))
        block_mean = tl.sum(values, axis=0) / visible_count
        centered = tl.where(mask, values - block_mean, 0.0)
        block_m2 = tl.sum(centered * centered, axis=0)
        delta = block_mean - welford_mean
        total_count = seen_count + visible_count
        welford_mean += delta * visible_count / total_count
        m2 += block_m2 + delta * delta * seen_count * visible_count / total_count
        seen_count = total_count

    mean = welford_mean
    variance = m2 / seen_count
    old_mean = tl.load(running_mean_ptr + channel)
    old_variance = tl.load(running_var_ptr + channel)
    unbiased_variance = variance * count.to(tl.float32) / (count.to(tl.float32) - 1.0)
    tl.store(running_mean_ptr + channel, (1.0 - momentum) * old_mean + momentum * mean)
    tl.store(
        running_var_ptr + channel,
        (1.0 - momentum) * old_variance + momentum * unbiased_variance,
    )

    inv_std = tl.rsqrt(variance + eps)
    if HAS_WEIGHT:
        weight = tl.load(weight_ptr + channel)
    else:
        weight = 1.0
    if HAS_BIAS:
        bias = tl.load(bias_ptr + channel)
    else:
        bias = 0.0

    for block_start in range(0, count, BLOCK_SIZE):
        positions = block_start + tl.arange(0, BLOCK_SIZE)
        mask = positions < count
        sequence_index = positions // sequence_length
        sequence_position = positions % sequence_length
        input_offsets = (
            sequence_index * input_batch_stride
            + channel * input_channel_stride
            + sequence_position * input_position_stride
        )
        output_offsets = (
            sequence_index * output_batch_stride
            + channel * output_channel_stride
            + sequence_position * output_position_stride
        )
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        normalized = (values - mean) * inv_std * weight + bias
        tl.store(output_ptr + output_offsets, tl.sigmoid(normalized), mask=mask)


def _make_eval_launcher(input, running_mean, running_var, weight, bias, eps):
    def launch():
        output = torch.empty_like(input)
        if input.ndim == 2 and input.stride(1) == 1:
            num_channels = input.shape[1]
            block_rows = 32
            block_channels = 256
            grid = (
                triton.cdiv(num_channels, block_channels),
                triton.cdiv(input.shape[0], block_rows),
            )
            _sigmoid_batch_norm_2d_kernel[grid](
                input,
                output,
                running_mean,
                running_var,
                weight if weight is not None else input,
                bias if bias is not None else input,
                num_channels,
                input.stride(0),
                output.stride(0),
                eps,
                HAS_WEIGHT=weight is not None,
                HAS_BIAS=bias is not None,
                BLOCK_ROWS=block_rows,
                BLOCK_CHANNELS=block_channels,
                num_warps=4,
            )
            return output

        channel_dimension = 1
        num_channels = input.shape[channel_dimension]
        sequence_length = input.shape[2] if input.ndim == 3 else 1
        batch_count = input.shape[0] * sequence_length
        block_size = 1024
        grid = (num_channels, triton.cdiv(batch_count, block_size))
        input_batch_stride = input.stride(0)
        input_channel_stride = input.stride(1)
        input_position_stride = input.stride(2) if input.ndim == 3 else 0
        output_batch_stride = output.stride(0)
        output_channel_stride = output.stride(1)
        output_position_stride = output.stride(2) if output.ndim == 3 else 0
        _sigmoid_batch_norm_eval_generic_kernel[grid](
            input,
            output,
            running_mean,
            running_var,
            weight if weight is not None else input,
            bias if bias is not None else input,
            batch_count,
            sequence_length,
            input_batch_stride,
            input_channel_stride,
            input_position_stride,
            output_batch_stride,
            output_channel_stride,
            output_position_stride,
            eps,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        return output

    return launch


def build(context):
    def sigmoid_batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-05,
    ):
        if input.ndim not in (2, 3):
            raise ValueError("input must have shape (N, C) or (N, C, L)")
        if input.shape[1] != running_mean.shape[0] or input.shape[1] != running_var.shape[0]:
            raise ValueError("running statistic size must equal the channel count")
        if weight is not None and weight.shape[0] != input.shape[1]:
            raise ValueError("weight size must equal the channel count")
        if bias is not None and bias.shape[0] != input.shape[1]:
            raise ValueError("bias size must equal the channel count")

        if not training:
            return _make_eval_launcher(input, running_mean, running_var, weight, bias, eps)()

        output = torch.empty_like(input)
        sequence_length = input.shape[2] if input.ndim == 3 else 1
        batch_count = input.shape[0]
        _sigmoid_batch_norm_training_kernel[(input.shape[1],)](
            input,
            output,
            running_mean,
            running_var,
            weight if weight is not None else input,
            bias if bias is not None else input,
            batch_count,
            sequence_length,
            input.stride(0),
            input.stride(1),
            input.stride(2) if input.ndim == 3 else 0,
            output.stride(0),
            output.stride(1),
            output.stride(2) if output.ndim == 3 else 0,
            momentum,
            eps,
            HAS_WEIGHT=weight is not None,
            HAS_BIAS=bias is not None,
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        return output

    return sigmoid_batch_norm
