from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _moment_kernel(
    input1_ptr,
    input2_ptr,
    stats_ptr,
    channels,
    spatial,
    num_groups,
    channels_per_group,
    tiles_per_channel,
    TILE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    tiles_per_group = channels_per_group * tiles_per_channel
    group_instance = tile_id // tiles_per_group
    tile_in_group = tile_id % tiles_per_group

    group_id = group_instance % num_groups

    channel_tile = tile_in_group // tiles_per_channel
    spatial_tile = tile_in_group % tiles_per_channel
    channel = group_id * channels_per_group + channel_tile
    base_offset = (group_instance * channels + channel) * spatial + spatial_tile * TILE
    offsets = base_offset + tl.arange(0, TILE)

    value1 = tl.load(input1_ptr + offsets)
    value2 = tl.load(input2_ptr + offsets)
    values = (value1 + value2) * value2

    mean = tl.sum(values, axis=0) / TILE
    centered = values - mean
    variance = tl.sum(centered * centered, axis=0) / TILE

    stats_base = tile_id * 2
    tl.store(stats_ptr + stats_base, mean)
    tl.store(stats_ptr + stats_base + 1, variance)


@triton.jit
def _normalize_kernel(
    input1_ptr,
    input2_ptr,
    stats_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    channels,
    spatial,
    num_groups,
    channels_per_group,
    tiles_per_channel,
    eps,
    TILE: tl.constexpr,
    GROUP_TILES: tl.constexpr,
):
    tile_id = tl.program_id(0)
    tiles_per_group = channels_per_group * tiles_per_channel
    group_instance = tile_id // tiles_per_group
    tile_in_group = tile_id % tiles_per_group
    channel_tile = tile_in_group // tiles_per_channel
    spatial_tile = tile_in_group % tiles_per_channel
    group_id = group_instance % num_groups
    channel = group_id * channels_per_group + channel_tile
    base_offset = (group_instance * channels + channel) * spatial + spatial_tile * TILE
    offsets = base_offset + tl.arange(0, TILE)

    reduction_ids = tl.arange(0, GROUP_TILES)
    stats_base = group_instance * tiles_per_group * 2 + reduction_ids * 2
    reduction_valid = reduction_ids < tiles_per_group
    partial_means = tl.load(stats_ptr + stats_base, mask=reduction_valid, other=0.0)
    partial_variances = tl.load(
        stats_ptr + stats_base + 1, mask=reduction_valid, other=0.0
    )
    mean = tl.sum(partial_means, axis=0) / tiles_per_group
    variance = (
        tl.sum(
            partial_variances
            + (partial_means - mean) * (partial_means - mean),
            axis=0,
        )
        / tiles_per_group
    )
    rstd = tl.rsqrt(variance + eps)

    value1 = tl.load(input1_ptr + offsets)
    value2 = tl.load(input2_ptr + offsets)
    values = (value1 + value2) * value2
    weight = tl.load(weight_ptr + channel)
    bias = tl.load(bias_ptr + channel)
    result = (values - mean) * rstd * weight + bias
    tl.store(output_ptr + offsets, result)


def build(context):
    def fused_add_mul_groupnorm(
        input1,
        input2,
        weight,
        bias,
        num_groups,
        eps=1e-05,
        *,
        out=None,
    ):
        if input1.ndim != 4 or input2.shape != input1.shape:
            raise ValueError("fixed-profile inputs must be matching 4-D tensors")
        if not input1.is_contiguous() or not input2.is_contiguous():
            raise ValueError("fixed-profile inputs must be contiguous")
        if weight.ndim != 1 or bias.ndim != 1:
            raise ValueError("weight and bias must be 1-D tensors")

        batch, channels = input1.shape[0], input1.shape[1]
        spatial = input1.shape[2] * input1.shape[3]
        if channels % num_groups != 0:
            raise ValueError("num_groups must divide the channel count")
        if weight.shape[0] != channels or bias.shape[0] != channels:
            raise ValueError("affine parameters must have one value per channel")

        tile = min(2048, triton.next_power_of_2(spatial))
        while spatial % tile != 0:
            tile //= 2
        tiles_per_channel = spatial // tile
        reduction = triton.next_power_of_2(channels_per_group * tiles_per_channel)
        channels_per_group = channels // num_groups

        if out is None:
            output = torch.empty_like(input1)
        else:
            if out.shape != input1.shape or out.dtype != input1.dtype:
                raise ValueError("out must match the input shape and dtype")
            if not out.is_contiguous():
                raise ValueError("fixed-profile out tensor must be contiguous")
            output = out
        stats = torch.empty(
            (batch * num_groups * channels_per_group * tiles_per_channel * 2,),
            device=input1.device,
            dtype=torch.float32,
        )
        grid = (batch * num_groups * channels_per_group * tiles_per_channel,)
        _moment_kernel[grid](
            input1,
            input2,
            stats,
            channels,
            spatial,
            num_groups,
            channels_per_group,
            tiles_per_channel,
            TILE=tile,
            num_warps=8,
        )
        _normalize_kernel[grid](
            input1,
            input2,
            stats,
            weight,
            bias,
            output,
            channels,
            spatial,
            num_groups,
            channels_per_group,
            tiles_per_channel,
            eps,
            TILE=tile,
            GROUP_TILES=reduction,
            num_warps=8,
        )
        return output

    return fused_add_mul_groupnorm
