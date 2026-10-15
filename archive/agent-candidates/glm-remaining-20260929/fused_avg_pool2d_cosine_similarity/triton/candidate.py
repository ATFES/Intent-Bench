from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def _cosine_kernel(
    x1_ptr,
    x2_ptr,
    cosine_ptr,
    channel_count,
    spatial_count,
    eps,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    batch_id = pid // ((spatial_count + BLOCK_HW - 1) // BLOCK_HW)
    tile_id = pid % ((spatial_count + BLOCK_HW - 1) // BLOCK_HW)
    hw = tile_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw < spatial_count

    dot = tl.zeros([BLOCK_HW], dtype=tl.float32)
    norm1_sq = tl.zeros([BLOCK_HW], dtype=tl.float32)
    norm2_sq = tl.zeros([BLOCK_HW], dtype=tl.float32)
    batch_offset = batch_id * channel_count * spatial_count
    for channel in range(0, channel_count):
        input_offset = batch_offset + channel * spatial_count + hw
        value1 = tl.load(x1_ptr + input_offset, mask=hw_mask, other=0.0)
        value2 = tl.load(x2_ptr + input_offset, mask=hw_mask, other=0.0)
        dot += value1 * value2
        norm1_sq += value1 * value1
        norm2_sq += value2 * value2

    dot_sum = tl.sum(dot, axis=0)
    norm1 = tl.sqrt(tl.sum(norm1_sq, axis=0))
    norm2 = tl.sqrt(tl.sum(norm2_sq, axis=0))
    norm1 = tl.maximum(norm1, eps)
    norm2 = tl.maximum(norm2, eps)
    result = dot_sum / (norm1 * norm2)
    tl.store(cosine_ptr + batch_id * spatial_count + hw, result, mask=hw_mask)


@triton.jit
def _avg_pool_kernel(
    cosine_ptr,
    output_ptr,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_size,
    stride,
    padding,
    BLOCK_OUT: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    output_tiles = (output_height * output_width + BLOCK_OUT - 1) // BLOCK_OUT
    batch_id = pid // output_tiles
    tile_id = pid % output_tiles
    output_index = tile_id * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    output_mask = output_index < output_height * output_width
    output_y = output_index // output_width
    output_x = output_index % output_width

    total = tl.zeros([BLOCK_OUT], dtype=tl.float32)
    batch_offset = batch_id * input_height * input_width
    for kernel_y in range(0, kernel_size):
        input_y = output_y * stride - padding + kernel_y
        row_mask = output_mask & (input_y >= 0) & (input_y < input_height)
        row_offset = batch_offset + input_y * input_width
        for kernel_x in range(0, kernel_size):
            input_x = output_x * stride - padding + kernel_x
            value_mask = row_mask & (input_x >= 0) & (input_x < input_width)
            input_offset = row_offset + input_x
            total += tl.load(cosine_ptr + input_offset, mask=value_mask, other=0.0)

    result = total / (kernel_size * kernel_size)
    tl.store(output_ptr + batch_id * output_height * output_width + output_index, result, mask=output_mask)


def build(context):
    def fused_avg_pool2d_cosine_similarity(
        x1: torch.Tensor,
        x2: torch.Tensor,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        if stride is None:
            stride = kernel_size

        batch_size, _, input_height, input_width = x1.shape
        output_height = (input_height + 2 * padding - kernel_size) // stride + 1
        output_width = (input_width + 2 * padding - kernel_size) // stride + 1
        spatial_count = input_height * input_width
        cosine = torch.empty(
            (batch_size, spatial_count),
            device=x1.device,
            dtype=torch.float32,
        )
        output = torch.empty(
            (batch_size, 1, output_height, output_width),
            device=x1.device,
            dtype=torch.float32,
        )

        block_hw = 256
        cosine_grid = (batch_size * ((spatial_count + block_hw - 1) // block_hw),)
        _cosine_kernel[cosine_grid](
            x1,
            x2,
            cosine,
            x1.shape[1],
            spatial_count,
            eps,
            BLOCK_HW=block_hw,
            num_warps=4,
        )

        block_out = 64
        output_tiles = (output_height * output_width + block_out - 1) // block_out
        pool_grid = (batch_size * output_tiles,)
        _avg_pool_kernel[pool_grid](
            cosine,
            output,
            input_height,
            input_width,
            output_height,
            output_width,
            kernel_size,
            stride,
            padding,
            BLOCK_OUT=block_out,
            num_warps=2,
        )
        return output

    return fused_avg_pool2d_cosine_similarity
