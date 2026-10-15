from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import pow as device_pow


@triton.jit
def _distance_kernel(
    x1_ptr,
    x2_ptr,
    distance_ptr,
    num_rows,
    feature_size,
    x1_row_stride,
    x1_vector_stride,
    x1_element_stride,
    x2_row_stride,
    x2_vector_stride,
    x2_element_stride,
    p_distance: tl.constexpr,
    eps_distance,
    block_size: tl.constexpr,
):
    row_id = tl.program_id(0)
    if row_id >= num_rows:
        return
    batch_id = row_id // feature_size
    vector_id = row_id % feature_size
    offsets = tl.arange(0, block_size)
    mask = offsets < feature_size
    x1_offsets = (
        batch_id * x1_row_stride
        + vector_id * x1_vector_stride
        + offsets * x1_element_stride
    )
    x2_offsets = (
        batch_id * x2_row_stride
        + vector_id * x2_vector_stride
        + offsets * x2_element_stride
    )
    difference = tl.load(x1_ptr + x1_offsets, mask=mask, other=0.0).to(tl.float32) - tl.load(
        x2_ptr + x2_offsets, mask=mask, other=0.0
    ).to(tl.float32)
    if p_distance == 2.0:
        norm_squared = tl.sum(difference * difference, axis=0)
        distance = tl.math.sqrt(norm_squared)
    else:
        norm_value = tl.sum(device_pow(tl.math.abs(difference), p_distance), axis=0)
        distance = device_pow(norm_value, 1.0 / p_distance)
    distance = tl.maximum(distance, eps_distance)
    tl.store(distance_ptr + row_id, distance)


@triton.jit
def _normalize_kernel(
    distance_ptr,
    output_ptr,
    norm_size,
    output_batch_stride,
    p_norm: tl.constexpr,
    eps_norm,
    block_size: tl.constexpr,
):
    batch_id = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < norm_size
    distance = tl.load(distance_ptr + batch_id * norm_size + offsets, mask=mask, other=0.0).to(
        tl.float32
    )
    if p_norm == 2:
        norm = tl.math.sqrt(tl.sum(distance * distance, axis=0))
    else:
        norm = device_pow(tl.sum(device_pow(tl.math.abs(distance), p_norm), axis=0), 1.0 / p_norm)
    scale = 1.0 / tl.maximum(norm, eps_norm)
    output_offsets = batch_id * output_batch_stride + offsets
    tl.store(output_ptr + output_offsets, distance * scale, mask=mask)


def _next_power_of_two(value: int) -> int:
    return max(1, triton.next_power_of_2(value))


def build(context):
    def normalize_pairwise_distance(
        x1,
        x2,
        p_distance=2.0,
        eps_distance=1e-6,
        keepdim=False,
        p_norm=2,
        dim_norm=1,
        eps_norm=1e-12,
    ):
        if x1.shape != x2.shape:
            raise ValueError("x1 and x2 must have the same shape")
        if x1.ndim != 3:
            raise ValueError("x1 and x2 must be three-dimensional")
        normalized_dim = dim_norm % x1.ndim
        if normalized_dim != 1:
            raise ValueError("dim_norm is out of range")

        batch_size, norm_size, feature_size = x1.shape
        distance = torch.empty(
            (batch_size, norm_size), dtype=torch.float32, device=x1.device
        )
        rows = batch_size * norm_size
        block_size = _next_power_of_two(feature_size)
        _distance_kernel[(rows,)](
            x1,
            x2,
            distance,
            rows,
            feature_size,
            x1.stride(0),
            x1.stride(1),
            x1.stride(2),
            x2.stride(0),
            x2.stride(1),
            x2.stride(2),
            p_distance=p_distance,
            eps_distance=eps_distance,
            block_size=block_size,
            num_warps=4,
        )

        if keepdim:
            output_shape = list(x1.shape)
            output_shape[normalized_dim] = 1
            output = distance.view(output_shape)
        else:
            output = distance
        norm_block_size = _next_power_of_two(norm_size)
        _normalize_kernel[(batch_size,)](
            distance,
            output,
            norm_size,
            output.stride(0),
            p_norm=p_norm,
            eps_norm=eps_norm,
            block_size=norm_block_size,
            num_warps=4,
        )
        return output

    return normalize_pairwise_distance
