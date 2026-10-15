from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _linear_split_k_kernel(
    input_ptr,
    weight_ptr,
    partial_ptr,
    row_stride,
    input_stride,
    weight_out_stride,
    weight_in_stride,
    partial_split_stride,
    num_rows,
    out_features,
    total_k,
    k_per_split,
    K_BLOCKS_PER_SPLIT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_mask = rows < num_rows
    col_mask = cols < out_features

    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, total_k)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for block_index in tl.static_range(K_BLOCKS_PER_SPLIT):
        offs_k = k_start + block_index * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_end

        input_offsets = rows[:, None] * row_stride + offs_k[None, :] * input_stride
        weight_offsets = cols[None, :] * weight_out_stride + offs_k[:, None] * weight_in_stride
        input_tile = tl.load(
            input_ptr + input_offsets,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        weight_tile = tl.load(
            weight_ptr + weight_offsets,
            mask=col_mask[None, :] & k_mask[:, None],
            other=0.0,
        )
        accumulator = tl.dot(
            input_tile,
            weight_tile,
            accumulator,
            input_precision="ieee",
        )

    partial_offsets = (
        pid_k * partial_split_stride
        + rows[:, None] * out_features
        + cols[None, :]
    )
    tl.store(
        partial_ptr + partial_offsets,
        accumulator,
        mask=row_mask[:, None] & col_mask[None, :],
    )


@triton.jit
def _relu_layer_norm_kernel(
    partial_ptr,
    bias_ptr,
    output_ptr,
    has_bias: tl.constexpr,
    partial_split_stride,
    num_rows,
    out_features,
    eps,
    SPLIT_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    partial_base = partial_ptr + row * out_features
    output_base = output_ptr + row * out_features

    sum_value = 0.0
    square_sum_value = 0.0
    split_offsets = tl.arange(0, SPLIT_K)
    for tile_start in range(0, out_features, BLOCK_N):
        cols = tile_start + tl.arange(0, BLOCK_N)
        col_mask = cols < out_features
        partial_offsets = (
            split_offsets[:, None] * partial_split_stride
            + cols[None, :]
        )
        partial_values = tl.load(
            partial_base + partial_offsets,
            mask=col_mask[None, :],
            other=0.0,
        )
        linear_value = tl.sum(partial_values, axis=0)
        if has_bias:
            linear_value += tl.load(bias_ptr + cols, mask=col_mask, other=0.0)
        activated_value = tl.maximum(linear_value, 0.0)
        sum_value += tl.sum(activated_value, axis=0)
        square_sum_value += tl.sum(activated_value * activated_value, axis=0)

    mean = sum_value / out_features
    variance = square_sum_value / out_features - mean * mean
    inverse_std = tl.rsqrt(variance + eps)

    for tile_start in range(0, out_features, BLOCK_N):
        cols = tile_start + tl.arange(0, BLOCK_N)
        col_mask = cols < out_features
        partial_offsets = (
            split_offsets[:, None] * partial_split_stride
            + cols[None, :]
        )
        partial_values = tl.load(
            partial_base + partial_offsets,
            mask=col_mask[None, :],
            other=0.0,
        )
        linear_value = tl.sum(partial_values, axis=0)
        if has_bias:
            linear_value += tl.load(bias_ptr + cols, mask=col_mask, other=0.0)
        activated_value = tl.maximum(linear_value, 0.0)
        normalized_value = (activated_value - mean) * inverse_std
        tl.store(output_base + cols, normalized_value, mask=col_mask)


def _normalized_size(normalized_shape, default_size):
    if normalized_shape is None:
        return (default_size,)
    if isinstance(normalized_shape, int):
        return (normalized_shape,)
    return tuple(int(size) for size in normalized_shape)


def build(context):
    def fused_layer_norm_relu_linear(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        normalized_shape: torch.Size = None,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> torch.Tensor:
        in_features = weight.shape[1]
        out_features = weight.shape[0]
        normalized_sizes = _normalized_size(normalized_shape, in_features)
        normalized_numel = 1
        for size in normalized_sizes:
            normalized_numel *= size
        if normalized_numel != out_features:
            raise ValueError("normalized_shape does not match the linear output shape")

        num_rows = input.numel() // in_features
        if input.shape[-1] != in_features or input.numel() % in_features != 0:
            raise ValueError("input and weight shapes are incompatible")
        if bias is not None and bias.shape[0] != out_features:
            raise ValueError("bias and weight shapes are incompatible")

        split_k = 8 if in_features >= 1024 and num_rows <= 64 else 1
        split_k = min(split_k, in_features)
        block_m = 16
        block_n = 64
        block_k = 64
        k_per_split = (in_features + split_k - 1) // split_k
        k_blocks_per_split = (k_per_split + block_k - 1) // block_k

        partial = torch.empty(
            (split_k, num_rows, out_features),
            device=input.device,
            dtype=torch.float32,
        )
        output = torch.empty(
            (*input.shape[:-1], out_features),
            device=input.device,
            dtype=input.dtype,
        )

        grid = (
            (num_rows + block_m - 1) // block_m,
            (out_features + block_n - 1) // block_n,
            split_k,
        )
        _linear_split_k_kernel[grid](
            input,
            weight,
            partial,
            in_features if input.dim() == 1 else input.stride(-2),
            input.stride(-1),
            weight.stride(0),
            weight.stride(1),
            num_rows * out_features,
            num_rows,
            out_features,
            in_features,
            k_per_split,
            k_blocks_per_split,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=3,
        )
        _relu_layer_norm_kernel[(num_rows,)](
            partial,
            input if bias is None else bias,
            output,
            bias is not None,
            num_rows * out_features,
            num_rows,
            out_features,
            eps,
            SPLIT_K=split_k,
            BLOCK_N=512,
            num_warps=4,
        )
        return output

    return fused_layer_norm_relu_linear
