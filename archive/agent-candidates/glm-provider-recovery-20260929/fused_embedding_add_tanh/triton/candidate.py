from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import pow as device_pow
from triton.language.extra.cuda.libdevice import tanh as device_tanh


@triton.jit
def _renorm_rows(
    weight_ptr,
    vocabulary: tl.constexpr,
    dimension: tl.constexpr,
    row_stride: tl.constexpr,
    column_stride: tl.constexpr,
    max_norm: tl.constexpr,
    norm_type: tl.constexpr,
    D_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    row_base = row * row_stride
    norm = tl.zeros((1,), dtype=tl.float32)

    for column_start in tl.range(0, dimension, D_BLOCK):
        columns = column_start + tl.arange(0, D_BLOCK).to(tl.int64)
        column_mask = columns < dimension
        values = tl.load(
            weight_ptr + row_base + columns * column_stride,
            mask=column_mask,
            other=0.0,
        )
        absolute = tl.abs(values)
        powers = tl.where(absolute == 0.0, 0.0, device_pow(absolute, norm_type))
        norm += tl.sum(powers)

    norm = device_pow(norm, 1.0 / norm_type)
    scale = tl.where(norm > max_norm, max_norm / norm, 1.0)

    for column_start in tl.range(0, dimension, D_BLOCK):
        columns = column_start + tl.arange(0, D_BLOCK).to(tl.int64)
        column_mask = columns < dimension
        pointers = weight_ptr + row_base + columns * column_stride
        values = tl.load(pointers, mask=column_mask, other=0.0)
        tl.store(pointers, values * scale, mask=column_mask)


@triton.jit
def _fused_embedding_add_tanh(
    indices_ptr,
    weight_ptr,
    other_ptr,
    output_ptr,
    row_count: tl.constexpr,
    dimension: tl.constexpr,
    rank: tl.constexpr,
    input_sizes: tl.constexpr,
    input_strides: tl.constexpr,
    other_strides: tl.constexpr,
    output_strides: tl.constexpr,
    weight_row_stride: tl.constexpr,
    weight_column_stride: tl.constexpr,
    ROW_BLOCK: tl.constexpr,
    D_BLOCK: tl.constexpr,
):
    row_start = tl.program_id(0).to(tl.int64) * ROW_BLOCK
    rows = row_start + tl.arange(0, ROW_BLOCK).to(tl.int64)
    row_valid = rows < row_count

    input_offsets = tl.zeros((ROW_BLOCK,), dtype=tl.int64)
    other_offsets = tl.zeros((ROW_BLOCK,), dtype=tl.int64)
    output_offsets = tl.zeros((ROW_BLOCK,), dtype=tl.int64)
    remaining = rows

    for axis in tl.static_range(rank):
        coordinate = remaining // input_sizes[axis]
        remaining = remaining % input_sizes[axis]
        input_offsets += coordinate * input_strides[axis]
        other_offsets += coordinate * other_strides[axis]
        output_offsets += coordinate * output_strides[axis]

    indices = tl.load(indices_ptr + input_offsets, mask=row_valid, other=0)
    weight_rows = indices.to(tl.int64) * weight_row_stride
    other_rows = other_offsets
    output_rows = output_offsets

    for column_start in tl.range(0, dimension, D_BLOCK):
        columns = column_start + tl.arange(0, D_BLOCK).to(tl.int64)
        valid = row_valid[:, None] & (columns < dimension)[None, :]
        embedded = tl.load(
            weight_ptr
            + weight_rows[:, None]
            + columns[None, :] * weight_column_stride,
            mask=valid,
            other=0.0,
        )
        added = tl.load(
            other_ptr
            + other_rows[:, None]
            + columns[None, :] * other_strides[rank],
            mask=valid,
            other=0.0,
        )
        result = device_tanh(embedded + added)
        tl.store(
            output_ptr
            + output_rows[:, None]
            + columns[None, :] * output_strides[rank],
            result,
            mask=valid,
        )


def _broadcast_strides(tensor: torch.Tensor, shape: tuple[int, ...]) -> tuple[int, ...]:
    tensor_shape = tensor.shape
    tensor_strides = tensor.stride()
    rank_difference = len(shape) - len(tensor_shape)
    result: list[int] = [0] * rank_difference

    for axis, size in enumerate(tensor_shape):
        output_axis = rank_difference + axis
        result.append(tensor_strides[axis] if size == shape[output_axis] else 0)

    return tuple(result)


def build(context):
    del context

    def fused_embedding_add_tanh(
        input_indices,
        weight,
        other,
        *,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        out=None,
    ):
        output_shape = tuple(input_indices.shape) + (weight.shape[1],)
        broadcast_shape = torch.broadcast_shapes(other.shape, output_shape)
        if broadcast_shape != output_shape:
            raise ValueError("other is not broadcastable to the embedding output shape")
        if out is None:
            output = torch.empty(
                output_shape,
                dtype=weight.dtype,
                device=weight.device,
            )
        else:
            if tuple(out.shape) != output_shape:
                raise ValueError("out must have the embedding output shape")
            output = out

        if max_norm is not None:
            if norm_type <= 0.0 or math.isinf(norm_type) or math.isnan(norm_type):
                raise ValueError("norm_type must be a finite positive number")
            _renorm_rows[(weight.shape[0],)](
                weight,
                vocabulary=weight.shape[0],
                dimension=weight.shape[1],
                row_stride=weight.stride(0),
                column_stride=weight.stride(1),
                max_norm=max_norm,
                norm_type=norm_type,
                D_BLOCK=128,
            )

        del padding_idx, scale_grad_by_freq, sparse
        input_sizes = tuple(input_indices.shape) if input_indices.ndim else ()
        input_strides = tuple(input_indices.stride()) if input_indices.ndim else ()
        other_strides = _broadcast_strides(other, output_shape)
        output_strides = tuple(output.stride())
        row_block = 16
        dimension_block = 128
        row_count = input_indices.numel()
        grid = (triton.cdiv(row_count, row_block),)

        if row_count == 0:
            return output

        _fused_embedding_add_tanh[grid](
            input_indices,
            weight,
            other,
            output,
            row_count=row_count,
            dimension=weight.shape[1],
            rank=input_indices.ndim,
            input_sizes=input_sizes,
            input_strides=input_strides,
            other_strides=other_strides,
            output_strides=output_strides,
            weight_row_stride=weight.stride(0),
            weight_column_stride=weight.stride(1),
            ROW_BLOCK=row_block,
            D_BLOCK=dimension_block,
        )
        return output

    return fused_embedding_add_tanh
