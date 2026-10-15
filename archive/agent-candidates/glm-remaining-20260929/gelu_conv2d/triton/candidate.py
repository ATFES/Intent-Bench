from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import torch
import triton
import triton.language as tl


def _pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    pair_value = tuple(value)
    if len(pair_value) != 2:
        raise ValueError("stride, padding, and dilation must contain two values")
    return int(pair_value[0]), int(pair_value[1])


def _padding_pair(
    padding: Union[int, Tuple[int, int], str],
    kernel: Tuple[int, int],
    dilation: Tuple[int, int],
) -> Tuple[int, int]:
    if isinstance(padding, str):
        mode = padding.lower()
        if mode == "valid":
            return 0, 0
        if mode == "same":
            extent_h = dilation[0] * (kernel[0] - 1)
            extent_w = dilation[1] * (kernel[1] - 1)
            if extent_h % 2 or extent_w % 2:
                raise ValueError("'same' padding requires an even effective kernel extent")
            return extent_h // 2, extent_w // 2
        raise ValueError(f"unsupported padding mode: {padding}")
    return _pair(padding)


@triton.jit
def _gelu_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    weight_output_stride,
    weight_input_stride,
    weight_height_stride,
    weight_width_stride,
    output_batch_stride,
    output_channel_stride,
    output_height_stride,
    output_width_stride,
    input_height,
    input_width,
    output_height,
    output_width,
    channels_per_group,
    outputs_per_group,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    KH: tl.constexpr,
    KW: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    TANH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_id = tl.program_id(0)
    output_tile_id = tl.program_id(1)
    batch = tl.program_id(2)
    tiles_per_group = (outputs_per_group + BLOCK_N - 1) // BLOCK_N
    group = output_tile_id // tiles_per_group
    local_output_tile = output_tile_id % tiles_per_group

    positions = tile_id * BLOCK_M + tl.arange(0, BLOCK_M)
    position_valid = positions < output_height * output_width
    row = positions // output_width
    column = positions % output_width
    input_row = row * stride_h - padding_h
    input_column = column * stride_w - padding_w

    channels = (
        group * outputs_per_group
        + local_output_tile * BLOCK_N
        + tl.arange(0, BLOCK_N)
    )
    channel_valid = channels < (group + 1) * outputs_per_group

    reduction_size = channels_per_group * KH * KW
    window_size = KH * KW
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for reduction_start in range(0, reduction_size, BLOCK_K):
        reduction = reduction_start + tl.arange(0, BLOCK_K)
        reduction_valid = reduction < reduction_size
        local_channel = reduction // window_size
        window = reduction % window_size
        window_row = window // KW
        window_column = window % KW

        source_row = input_row[:, None] + window_row[None, :] * dilation_h
        source_column = input_column[:, None] + window_column[None, :] * dilation_w
        source_channel = group * channels_per_group + local_channel[None, :]
        row_valid = (source_row >= 0) & (source_row < input_height)
        column_valid = (source_column >= 0) & (source_column < input_width)
        input_valid = (
            position_valid[:, None]
            & reduction_valid[None, :]
            & row_valid
            & column_valid
        )
        input_offsets = (
            batch * input_batch_stride
            + source_channel * input_channel_stride
            + source_row * input_height_stride
            + source_column * input_width_stride
        )
        input_tile = tl.load(input_ptr + input_offsets, mask=input_valid, other=0.0)

        weight_offsets = (
            channels[None, :] * weight_output_stride
            + local_channel[:, None] * weight_input_stride
            + window_row[:, None] * weight_height_stride
            + window_column[:, None] * weight_width_stride
        )
        weight_tile = tl.load(
            weight_ptr + weight_offsets,
            mask=reduction_valid[:, None] & channel_valid[None, :],
            other=0.0,
        )
        accumulator = tl.dot(
            input_tile,
            weight_tile,
            acc=accumulator,
            input_precision="ieee",
            out_dtype=tl.float32,
        )

    if HAS_BIAS:
        bias = tl.load(bias_ptr + channels, mask=channel_valid, other=0.0)
        accumulator += bias[None, :]

    if TANH:
        inner = 0.7978845608028654 * (
            accumulator + 0.044715 * accumulator * accumulator * accumulator
        )
        exponential = tl.exp(-2.0 * inner)
        activated = 0.5 * accumulator * (2.0 / (1.0 + exponential))
    else:
        normal = accumulator * 0.7071067811865476
        activated = 0.5 * accumulator * (1.0 + tl.erf(normal))

    output_offsets = (
        batch * output_batch_stride
        + channels[None, :] * output_channel_stride
        + row[:, None] * output_height_stride
        + column[:, None] * output_width_stride
    )
    tl.store(
        output_ptr + output_offsets,
        activated,
        mask=position_valid[:, None] & channel_valid[None, :],
    )


def build(context) -> Callable[
    [
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Union[int, Tuple[int, int]],
        Union[int, Tuple[int, int], str],
        Union[int, Tuple[int, int]],
        int,
        str,
        Optional[torch.Tensor],
    ],
    torch.Tensor,
]:
    def gelu_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        approximate: str = "none",
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input.ndim != 4 or weight.ndim != 4:
            raise ValueError("input and weight must be four-dimensional")
        if input.device != weight.device:
            raise ValueError("input and weight must be on the same device")
        if input.device.type != "cuda":
            raise ValueError("input and weight must be CUDA tensors")
        if input.dtype != torch.float32 or weight.dtype != torch.float32:
            raise ValueError("input and weight must be float32")
        if approximate not in ("none", "tanh"):
            raise ValueError("approximate must be 'none' or 'tanh'")

        stride_h, stride_w = _pair(stride)
        dilation_h, dilation_w = _pair(dilation)
        padding_h, padding_w = _padding_pair(
            padding, (weight.shape[2], weight.shape[3]), (dilation_h, dilation_w)
        )
        if stride_h <= 0 or stride_w <= 0 or dilation_h <= 0 or dilation_w <= 0:
            raise ValueError("stride and dilation values must be positive")
        if padding_h < 0 or padding_w < 0:
            raise ValueError("padding values must be non-negative")
        if groups <= 0 or input.shape[1] % groups:
            raise ValueError("incompatible input channels for groups")

        batch_size, in_channels, input_height, input_width = input.shape
        out_channels, channels_per_group, kernel_height, kernel_width = weight.shape
        if out_channels % groups or channels_per_group != in_channels // groups:
            raise ValueError("incompatible channel shape for groups")
        output_height = (
            input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1
        ) // stride_h + 1
        output_width = (
            input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1
        ) // stride_w + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("convolution output is empty")

        has_bias = bias is not None
        if has_bias:
            if bias.device != input.device or bias.dtype != torch.float32:
                raise ValueError("bias must be a CUDA float32 tensor")
            if bias.shape != (out_channels,):
                raise ValueError("bias shape must equal out_channels")
            bias_argument = bias
        else:
            bias_argument = weight

        if out is None:
            output = torch.empty(
                (batch_size, out_channels, output_height, output_width),
                dtype=torch.float32,
                device=input.device,
            )
        else:
            if out.device != input.device or out.dtype != torch.float32:
                raise ValueError("out must be a CUDA float32 tensor")
            if tuple(out.shape) != (batch_size, out_channels, output_height, output_width):
                raise ValueError("out has an incompatible shape")
            output = out

        block_m = 64
        block_n = 32 if out_channels >= 32 else 16
        block_k = 32
        output_tiles_per_group = triton.cdiv(out_channels // groups, block_n)
        grid = (
            triton.cdiv(output_height * output_width, block_m),
            groups * output_tiles_per_group,
            batch_size,
        )
        _gelu_conv2d_kernel[grid](
            input,
            weight,
            bias_argument,
            output,
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
            input_height,
            input_width,
            output_height,
            output_width,
            channels_per_group,
            out_channels // groups,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            KH=kernel_height,
            KW=kernel_width,
            HAS_BIAS=has_bias,
            TANH=approximate == "tanh",
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )
        return output

    return gelu_conv2d
