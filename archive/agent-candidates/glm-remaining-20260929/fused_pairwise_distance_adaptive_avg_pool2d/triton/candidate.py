from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _pool_row_norm_partials_kernel(
    x1_ptr,
    x2_ptr,
    partials_ptr,
    input_height,
    input_width,
    output_width,
    x1_batch_stride,
    x1_channel_stride,
    x1_height_stride,
    x1_width_stride,
    x2_batch_stride,
    x2_channel_stride,
    x2_height_stride,
    x2_width_stride,
    eps,
    channels: tl.constexpr,
    channel_spatial_block: tl.constexpr,
):
    output_row = tl.program_id(0)
    width_partition = tl.program_id(1)
    batch = tl.program_id(2)
    lane = tl.arange(0, channel_spatial_block)
    lane_valid = lane < channels * output_width
    channel = lane // output_width
    output_col = width_partition * output_width + lane % output_width
    difference_sum = tl.zeros((channel_spatial_block,), dtype=tl.float32)

    for height_offset in tl.static_range(2):
        input_row = 2 * output_row + height_offset
        for width_offset in tl.static_range(2):
            input_col = 2 * output_col + width_offset
            x1_offset = (
                batch * x1_batch_stride
                + channel * x1_channel_stride
                + input_row * x1_height_stride
                + input_col * x1_width_stride
            )
            x2_offset = (
                batch * x2_batch_stride
                + channel * x2_channel_stride
                + input_row * x2_height_stride
                + input_col * x2_width_stride
            )
            x1_value = tl.load(x1_ptr + x1_offset, mask=lane_valid, other=0.0)
            x2_value = tl.load(x2_ptr + x2_offset, mask=lane_valid, other=0.0)
            difference_sum += x1_value - x2_value

    pooled_difference = difference_sum * 0.25 + eps
    norm_partials = pooled_difference * pooled_difference

    partial_index = (
        batch * tl.num_programs(0) * tl.num_programs(1)
        + output_row * tl.num_programs(1)
        + width_partition
    )
    tl.store(partials_ptr + partial_index, tl.sum(norm_partials, axis=0))


@triton.jit
def _pool_pixel_norm_partials_kernel(
    x1_ptr,
    x2_ptr,
    partials_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    x1_batch_stride,
    x1_channel_stride,
    x1_height_stride,
    x1_width_stride,
    x2_batch_stride,
    x2_channel_stride,
    x2_height_stride,
    x2_width_stride,
    norm_degree,
    eps,
    channels: tl.constexpr,
    channel_block: tl.constexpr,
    is_euclidean: tl.constexpr,
):
    program_id = tl.program_id(0)
    output_plane = output_height * output_width
    batch = program_id // output_plane
    plane_index = program_id % output_plane
    output_row = plane_index // output_width
    output_col = plane_index % output_width
    channel = tl.arange(0, channel_block)
    channel_valid = channel < channels
    difference_sum = tl.zeros((channel_block,), dtype=tl.float32)
    row_start = (output_row * input_height) // output_height
    row_end = ((output_row + 1) * input_height - 1) // output_height + 1
    col_start = (output_col * input_width) // output_width
    col_end = ((output_col + 1) * input_width - 1) // output_width + 1
    window_size = (row_end - row_start) * (col_end - col_start)

    for input_row in range(row_start, row_end):
        for input_col in range(col_start, col_end):
            x1_offset = (
                batch * x1_batch_stride
                + channel * x1_channel_stride
                + input_row * x1_height_stride
                + input_col * x1_width_stride
            )
            x2_offset = (
                batch * x2_batch_stride
                + channel * x2_channel_stride
                + input_row * x2_height_stride
                + input_col * x2_width_stride
            )
            x1_value = tl.load(x1_ptr + x1_offset, mask=channel_valid, other=0.0)
            x2_value = tl.load(x2_ptr + x2_offset, mask=channel_valid, other=0.0)
            difference_sum += x1_value - x2_value

    pooled_difference = difference_sum / window_size.to(tl.float32) + eps
    if is_euclidean:
        norm_partial = tl.sum(pooled_difference * pooled_difference, axis=0)
    else:
        norm_partial = tl.sum(tl.abs(pooled_difference) ** norm_degree, axis=0)
    tl.store(partials_ptr + program_id, norm_partial)


@triton.jit
def _reduce_norm_partials_kernel(
    partials_ptr,
    output_ptr,
    partial_count,
    norm_degree,
    partial_block: tl.constexpr,
    is_euclidean: tl.constexpr,
):
    batch = tl.program_id(0)
    partials = tl.arange(0, partial_block)
    norm_sum = tl.zeros((partial_block,), dtype=tl.float32)
    for start in range(0, partial_count, partial_block):
        offsets = start + partials
        values = tl.load(
            partials_ptr + batch * partial_count + offsets,
            mask=offsets < partial_count,
            other=0.0,
        )
        norm_sum += values

    if is_euclidean:
        distance = tl.sqrt_rn(norm_sum)
    else:
        distance = tl.sum(norm_sum, axis=0) ** (1.0 / norm_degree)
    tl.store(output_ptr + batch, distance)


def _normalize_output_size(output_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(output_size, int):
        height = width = output_size
    else:
        resolved_size = tuple(output_size)
        if len(resolved_size) != 2:
            raise ValueError("output_size must be an int or a pair of ints")
        height, width = resolved_size
    if height <= 0 or width <= 0:
        raise ValueError("output_size entries must be positive")
    return int(height), int(width)


def build(context):

    def fused_pairwise_distance_adaptive_avg_pool2d(
        x1: torch.Tensor,
        x2: torch.Tensor,
        output_size: int | tuple[int, int],
        p: float = 2.0,
        eps: float = 1e-06,
        keepdim: bool = False,
    ) -> torch.Tensor:
        if x1.ndim != 4 or x2.ndim != 4:
            raise ValueError("x1 and x2 must be four-dimensional NCHW tensors")
        if x1.shape != x2.shape:
            raise ValueError("x1 and x2 must have identical shapes")
        if x1.device != x2.device or x1.device.type != "cuda":
            raise ValueError("x1 and x2 must be CUDA tensors on the same device")
        if x1.dtype != torch.float32 or x2.dtype != torch.float32:
            raise ValueError("x1 and x2 must have torch.float32 dtype")

        batch_size, channel_count, input_height, input_width = x1.shape
        output_height, output_width = _normalize_output_size(output_size)
        is_euclidean = float(p) == 2.0
        is_two_times_downsample = (
            input_height == 2 * output_height and input_width == 2 * output_width
        )
        channel_spatial_count = channel_count * output_width
        uses_row_blocks = is_two_times_downsample and channel_spatial_count <= 1024

        if uses_row_blocks:
            channel_spatial_block = 1 << (channel_spatial_count - 1).bit_length()
            width_partitions = 1
            partial_count = output_height * width_partitions
            partials = torch.empty(
                (batch_size, partial_count),
                dtype=torch.float32,
                device=x1.device,
            )
            _pool_row_norm_partials_kernel[(output_height, width_partitions, batch_size)](
                x1,
                x2,
                partials,
                input_height,
                input_width,
                output_width,
                x1.stride(0),
                x1.stride(1),
                x1.stride(2),
                x1.stride(3),
                x2.stride(0),
                x2.stride(1),
                x2.stride(2),
                x2.stride(3),
                eps,
                channels=channel_count,
                channel_spatial_block=channel_spatial_block,
                num_warps=2,
            )
        else:
            partial_count = output_height * output_width
            channel_block = 1 << (channel_count - 1).bit_length()
            partials = torch.empty(
                (batch_size, partial_count),
                dtype=torch.float32,
                device=x1.device,
            )
            _pool_pixel_norm_partials_kernel[(batch_size * partial_count,)](
                x1,
                x2,
                partials,
                input_height,
                input_width,
                output_height,
                output_width,
                x1.stride(0),
                x1.stride(1),
                x1.stride(2),
                x1.stride(3),
                x2.stride(0),
                x2.stride(1),
                x2.stride(2),
                x2.stride(3),
                float(p),
                eps,
                channels=channel_count,
                channel_block=channel_block,
                is_euclidean=is_euclidean,
            )

        output = torch.empty((batch_size,), dtype=torch.float32, device=x1.device)
        _reduce_norm_partials_kernel[(batch_size,)](
            partials,
            output,
            partial_count,
            float(p),
            partial_block=256,
            is_euclidean=is_euclidean,
        )

        if keepdim:
            return output.view(batch_size, 1)
        return output

    return fused_pairwise_distance_adaptive_avg_pool2d
