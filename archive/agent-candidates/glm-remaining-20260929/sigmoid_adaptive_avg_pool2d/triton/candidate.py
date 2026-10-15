from __future__ import annotations

from typing import Callable, Tuple, Union

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_adaptive_avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    input_stride_n,
    input_stride_c,
    input_stride_h,
    input_stride_w,
    input_height: tl.constexpr,
    input_width: tl.constexpr,
    output_height: tl.constexpr,
    output_width: tl.constexpr,
    block_height: tl.constexpr,
    block_width: tl.constexpr,
):
    output_index = tl.program_id(0)
    output_y = output_index % output_height
    output_x = output_index // output_height
    channel = tl.program_id(1)
    batch = tl.program_id(2)

    window_start_y = (output_y * input_height) // output_height
    window_end_y = ((output_y + 1) * input_height + output_height - 1) // output_height
    window_start_x = (output_x * input_width) // output_width
    window_end_x = ((output_x + 1) * input_width + output_width - 1) // output_width

    window_y = window_start_y + tl.arange(0, block_height)
    window_x = window_start_x + tl.arange(0, block_width)
    window_mask = (window_y[:, None] < window_end_y) & (
        window_x[None, :] < window_end_x
    )
    input_offset = (
        batch * input_stride_n
        + channel * input_stride_c
        + window_y[:, None] * input_stride_h
        + window_x[None, :] * input_stride_w
    )
    window = tl.load(input_ptr + input_offset, mask=window_mask, other=0.0)
    pooled = tl.sum(tl.sum(window, axis=1), axis=0) / (
        (window_end_y - window_start_y) * (window_end_x - window_start_x)
    )
    activated = 1.0 / (1.0 + tl.exp(-pooled))
    tl.store(
        output_ptr
        + (batch * tl.num_programs(1) + channel) * output_width
        + output_x,
        activated,
    )


def _normalise_output_size(output_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(output_size, int):
        output_height = output_size
        output_width = output_size
    else:
        output_height, output_width = output_size
    if (
        not isinstance(output_height, int)
        or not isinstance(output_width, int)
        or output_height <= 0
        or output_width <= 0
    ):
        raise ValueError("output_size must contain positive integers")
    return output_height, output_width


def _next_power_of_two(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


def build(context) -> Callable[[torch.Tensor, Union[int, Tuple[int, int]]], torch.Tensor]:
    del context

    def sigmoid_adaptive_avg_pool2d(
        input: torch.Tensor, output_size: Union[int, Tuple[int, int]]
    ) -> torch.Tensor:
        if input.ndim != 4:
            raise ValueError("input must be a 4-D NCHW tensor")
        if not input.is_cuda or input.dtype != torch.float32:
            raise ValueError("input must be a CUDA float32 tensor")

        output_height, output_width = _normalise_output_size(output_size)
        batch_size, channels, input_height, input_width = input.shape
        output = torch.empty(
            (batch_size, channels, output_height, output_width),
            dtype=input.dtype,
            device=input.device,
        )

        block_height = min(
            input_height,
            _next_power_of_two(
                max(
                    1,
                    (input_height + output_height - 1) // output_height,
                )
            ),
        )
        block_width = min(
            input_width,
            _next_power_of_two(
                max(
                    1,
                    (input_width + output_width - 1) // output_width,
                )
            ),
        )

        _sigmoid_adaptive_avg_pool2d_kernel[(output_height * output_width, channels, batch_size)](
            input,
            output,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            input.stride(3),
            input_height=input_height,
            input_width=input_width,
            output_height=output_height,
            output_width=output_width,
            block_height=block_height,
            block_width=block_width,
        )
        return output

    return sigmoid_adaptive_avg_pool2d
