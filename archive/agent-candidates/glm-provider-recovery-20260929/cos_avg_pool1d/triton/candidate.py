from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cos_avg_pool1d_kernel(
    input_ptr,
    output_ptr,
    input_stride_b,
    input_stride_c,
    input_stride_w,
    input_length,
    output_length,
    output_stride_bc,
    channels,
    kernel_size: tl.constexpr,
    stride,
    padding,
    count_include_pad: tl.constexpr,
    block_size: tl.constexpr,
):
    output_offset = tl.program_id(0) * block_size + tl.arange(0, block_size)
    output_valid = output_offset < output_length
    batch_channel = tl.program_id(1)
    channel = batch_channel % channels
    batch = batch_channel // channels
    input_base = batch * input_stride_b + channel * input_stride_c

    accumulator = tl.zeros([block_size], dtype=tl.float32)
    valid_count = tl.zeros([block_size], dtype=tl.int32)
    window_start = output_offset * stride - padding

    for kernel_offset in tl.static_range(kernel_size):
        input_offset = window_start + kernel_offset
        input_valid = output_valid & (input_offset >= 0) & (input_offset < input_length)
        value = tl.load(
            input_ptr + input_base + input_offset * input_stride_w,
            mask=input_valid,
            other=0.0,
        )
        accumulator += tl.cos(value)
        valid_count += input_valid.to(tl.int32)

    if count_include_pad:
        denominator = kernel_size
    else:
        denominator = tl.maximum(valid_count, 1)

    result = accumulator / denominator
    output_offset_bc = batch_channel * output_stride_bc + output_offset
    tl.store(output_ptr + output_offset_bc, result, mask=output_valid)


def build(context):
    def cos_avg_pool1d(
        input: torch.Tensor,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> torch.Tensor:
        if stride is None:
            stride = kernel_size

        input_length = input.shape[-1]
        padded_length = input_length + 2 * padding
        if ceil_mode:
            output_length = (padded_length - kernel_size + stride - 1) // stride + 1
            if (output_length - 1) * stride >= input_length + padding:
                output_length -= 1
        else:
            output_length = (padded_length - kernel_size) // stride + 1

        output = torch.empty(
            (*input.shape[:-1], output_length),
            dtype=input.dtype,
            device=input.device,
        )
        if output_length <= 0 or input.numel() == 0:
            return output

        batch_count = input.shape[0]
        channels = input.shape[1]
        _cos_avg_pool1d_kernel[(triton.cdiv(output_length, 128), batch_count)](
            input,
            output,
            input.stride(-3),
            input.stride(-2),
            input.stride(-1),
            input_length,
            output_length,
            output_length,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            count_include_pad=count_include_pad,
            block_size=128,
        )
        return output

    return cos_avg_pool1d
