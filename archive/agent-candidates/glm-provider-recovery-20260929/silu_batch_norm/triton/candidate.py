from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _silu_batch_norm_eval_kernel(
    input_ptr,
    output_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    channels,
    spatial_size,
    eps,
    has_affine: tl.constexpr,
    block_size: tl.constexpr,
):
    plane_id = tl.program_id(0)
    batch_index = plane_id // channels
    channel_index = plane_id % channels

    channel_base = (batch_index * channels + channel_index) * spatial_size
    mean = tl.load(running_mean_ptr + channel_index)
    variance = tl.load(running_var_ptr + channel_index)
    inverse_std = tl.rsqrt(variance + eps)

    if has_affine:
        weight = tl.load(weight_ptr + channel_index)
        bias = tl.load(bias_ptr + channel_index)

    for start in range(0, spatial_size, block_size):
        offsets = start + tl.arange(0, block_size)
        mask = offsets < spatial_size
        input_offsets = channel_base + offsets
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        normalized = (values - mean) * inverse_std
        if has_affine:
            normalized = normalized * weight + bias
        output = normalized * tl.sigmoid(normalized)
        tl.store(output_ptr + input_offsets, output, mask=mask)


@triton.jit
def _batch_norm_stats_kernel(
    input_ptr,
    running_mean_ptr,
    running_var_ptr,
    batch_mean_ptr,
    batch_inverse_std_ptr,
    channels,
    inner_size,
    momentum,
    eps,
    block_size: tl.constexpr,
):
    channel_index = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    value_sum = tl.zeros([block_size], dtype=tl.float32)
    squared_sum = tl.zeros([block_size], dtype=tl.float32)

    for start in range(0, inner_size, block_size):
        input_offsets = channel_index * inner_size + start + offsets
        mask = start + offsets < inner_size
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        value_sum += values
        squared_sum += values * values

    count = inner_size.to(tl.float32)
    mean = tl.sum(value_sum, axis=0) / count
    variance = tl.sum(squared_sum, axis=0) / count - mean * mean
    variance = tl.maximum(variance, 0.0)
    inverse_std = tl.rsqrt(variance + eps)
    unbiased_variance = variance * count / (count - 1.0)

    old_mean = tl.load(running_mean_ptr + channel_index)
    old_variance = tl.load(running_var_ptr + channel_index)
    new_mean = (1.0 - momentum) * old_mean + momentum * mean
    new_variance = (1.0 - momentum) * old_variance + momentum * unbiased_variance

    tl.store(batch_mean_ptr + channel_index, mean)
    tl.store(batch_inverse_std_ptr + channel_index, inverse_std)
    tl.store(running_mean_ptr + channel_index, new_mean)
    tl.store(running_var_ptr + channel_index, new_variance)


@triton.jit
def _silu_batch_norm_train_kernel(
    input_ptr,
    output_ptr,
    batch_mean_ptr,
    batch_inverse_std_ptr,
    weight_ptr,
    bias_ptr,
    channels,
    spatial_size,
    has_affine: tl.constexpr,
    block_size: tl.constexpr,
):
    plane_id = tl.program_id(0)
    batch_index = plane_id // channels
    channel_index = plane_id % channels
    channel_base = (batch_index * channels + channel_index) * spatial_size

    mean = tl.load(batch_mean_ptr + channel_index)
    inverse_std = tl.load(batch_inverse_std_ptr + channel_index)
    if has_affine:
        weight = tl.load(weight_ptr + channel_index)
        bias = tl.load(bias_ptr + channel_index)
        scale = weight * inverse_std
        shift = bias - mean * scale
    else:
        scale = inverse_std
        shift = -mean * inverse_std

    for start in range(0, spatial_size, block_size):
        offsets = start + tl.arange(0, block_size)
        mask = offsets < spatial_size
        input_offsets = channel_base + offsets
        values = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        normalized = values * scale + shift
        output = normalized * tl.sigmoid(normalized)
        tl.store(output_ptr + input_offsets, output, mask=mask)


def build(context):
    del context

    def silu_batch_norm(
        input: torch.Tensor,
        running_mean: torch.Tensor,
        running_var: torch.Tensor,
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        training: bool = False,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        if input.ndim != 4:
            raise ValueError("input must be a 4-D NCHW tensor")
        if not input.is_cuda or not input.is_contiguous():
            raise ValueError("input must be a contiguous CUDA tensor")
        channels = input.shape[1]
        spatial_size = input.shape[2] * input.shape[3]
        output = torch.empty_like(input, memory_format=torch.contiguous_format)

        if not training:
            grid = (input.shape[0] * channels,)
            _silu_batch_norm_eval_kernel[grid](
                input,
                output,
                running_mean,
                running_var,
                weight if weight is not None else input,
                bias if bias is not None else input,
                channels,
                spatial_size,
                eps,
                has_affine=weight is not None,
                block_size=1024,
                num_warps=4,
            )
            return output

        if input.shape[0] * spatial_size < 2:
            raise ValueError("training requires at least two batch samples")
        if (weight is None) != (bias is None):
            raise ValueError("weight and bias must be supplied together")

        batch_mean = torch.empty_like(running_mean)
        batch_inverse_std = torch.empty_like(running_var)
        _batch_norm_stats_kernel[(channels,)](
            input,
            running_mean,
            running_var,
            batch_mean,
            batch_inverse_std,
            input.shape[0] * spatial_size,
            momentum,
            eps,
            block_size=512,
            num_warps=4,
        )
        _silu_batch_norm_train_kernel[(input.shape[0] * channels,)](
            input,
            output,
            batch_mean,
            batch_inverse_std,
            weight if weight is not None else input,
            bias if bias is not None else input,
            channels,
            spatial_size,
            has_affine=weight is not None,
            block_size=1024,
            num_warps=4,
        )
        return output

    return silu_batch_norm
