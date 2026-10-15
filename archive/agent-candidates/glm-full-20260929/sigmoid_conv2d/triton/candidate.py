from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    input_height,
    input_width,
    out_channels,
    output_height,
    output_width,
    input_batch_stride,
    input_channel_stride,
    input_row_stride,
    input_column_stride,
    weight_output_stride,
    weight_channel_stride,
    weight_row_stride,
    weight_column_stride,
    output_batch_stride,
    output_channel_stride,
    output_row_stride,
    output_column_stride,
    bias_stride,
    groups,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    kernel_h,
    kernel_w,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    output_pixels = output_height * output_width
    pixel_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    pixel_mask = pixel_offset < output_pixels
    output_y = pixel_offset // output_width
    output_x = pixel_offset % output_width
    input_y = output_y * stride_h - padding_h
    input_x = output_x * stride_w - padding_w

    channels_per_group = in_channels // groups
    outputs_per_group = out_channels // groups
    output_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    output_n_mask = output_n < out_channels
    group_id = (pid_n * BLOCK_N) // outputs_per_group
    first_group_channel = group_id * channels_per_group

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    channel = tl.arange(0, BLOCK_C)
    for channel_start in range(0, channels_per_group, BLOCK_C):
        channel_index = channel_start + channel
        channel_valid = channel_index < channels_per_group
        global_channel = first_group_channel + channel_index
        for kernel_y in range(0, kernel_h):
            source_y = input_y[:, None] + kernel_y * dilation_h
            row_valid = (source_y >= 0) & (source_y < input_height)
            for kernel_x in range(0, kernel_w):
                source_x = input_x[:, None] + kernel_x * dilation_w
                column_valid = (source_x >= 0) & (source_x < input_width)
                valid = (
                    pixel_mask[:, None]
                    & channel_valid[None, :]
                    & row_valid
                    & column_valid
                )
                input_offsets = (
                    pid_b * input_batch_stride
                    + global_channel * input_channel_stride
                    + source_y * input_row_stride
                    + source_x * input_column_stride
                )
                input_block = tl.load(
                    input_ptr + input_offsets,
                    mask=valid,
                    other=0.0,
                )

                weight_offsets = (
                    output_n[None, :] * weight_output_stride
                    + channel_index[:, None] * weight_channel_stride
                    + kernel_y * weight_row_stride
                    + kernel_x * weight_column_stride
                )
                weight_valid = (
                    output_n_mask[None, :] & channel_valid[:, None]
                )
                weight_block = tl.load(
                    weight_ptr + weight_offsets,
                    mask=weight_valid,
                    other=0.0,
                )
                accumulator = tl.dot(
                    input_block,
                    weight_block,
                    accumulator,
                    input_precision="ieee",
                )

    if HAS_BIAS:
        bias = tl.load(bias_ptr + output_n * bias_stride, mask=output_n_mask, other=0.0)
        accumulator += bias[None, :]

    result = tl.sigmoid(accumulator)
    output_offsets = (
        pid_b * output_batch_stride
        + output_n[None, :] * output_channel_stride
        + output_y[:, None] * output_row_stride
        + output_x[:, None] * output_column_stride
    )
    tl.store(
        output_ptr + output_offsets,
        result.to(output_ptr.dtype.element_ty),
        mask=pixel_mask[:, None] & output_n_mask[None, :],
    )


def _pair(value, name):
    if isinstance(value, str):
        if value == "valid":
            return 0, 0
        if value == "same":
            return None, None
        raise ValueError(f"invalid {name}: {value!r}")
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"invalid {name}: {value!r}")
        return int(value[0]), int(value[1])
    raise TypeError(f"{name} must be an int, tuple, or string")


def _build_sigmoid_conv2d(context):
    def sigmoid_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        out=None,
    ):
        if input.ndim != 4 or weight.ndim != 4:
            raise ValueError("sigmoid_conv2d expects 4D input and weight tensors")
        if bias is not None and bias.ndim != 1:
            raise ValueError("bias must be a 1D tensor")

        batch_size, in_channels, input_height, input_width = input.shape
        out_channels, weight_channels, kernel_h, kernel_w = weight.shape
        outputs_per_group = out_channels // groups
        stride_h, stride_w = _pair(stride, "stride")
        dilation_h, dilation_w = _pair(dilation, "dilation")
        padding_h, padding_w = _pair(padding, "padding")
        if padding_h is None or padding_w is None:
            if stride_h != 1 or stride_w != 1 or dilation_h != 1 or dilation_w != 1:
                raise ValueError("padding='same' requires unit stride and dilation")
            padding_h = (kernel_h - 1) // 2
            padding_w = (kernel_w - 1) // 2

        output_height = (
            input_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        output_width = (
            input_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("convolution output is empty")

        if out is None:
            output = torch.empty(
                (batch_size, out_channels, output_height, output_width),
                device=input.device,
                dtype=input.dtype,
            )
        else:
            output = out

        input_elements = input.numel()
        weight_elements = weight.numel()
        output_elements = output.numel()
        if max(input_elements, weight_elements, output_elements) >= 2**31:
            raise ValueError("sigmoid_conv2d Triton kernel requires tensors below 2G elements")
        block_n = 64
        if (
            groups < 1
            or in_channels % groups != 0
            or out_channels % groups != 0
            or weight_channels != in_channels // groups
            or (groups != 1 and outputs_per_group % block_n != 0)
        ):
            raise ValueError("invalid convolution dimensions or unsupported grouped-output tiling")

        bias_pointer = weight if bias is None else bias
        _sigmoid_conv2d_kernel[(
            triton.cdiv(output_height * output_width, 32),
            triton.cdiv(out_channels, block_n),
            batch_size,
        )](
            input,
            weight,
            bias_pointer,
            output,
            batch_size,
            in_channels,
            input_height,
            input_width,
            out_channels,
            output_height,
            output_width,
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
            1 if bias is None else bias.stride(0),
            groups,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            kernel_h,
            kernel_w,
            HAS_BIAS=bias is not None,
            BLOCK_M=32,
            BLOCK_N=block_n,
            BLOCK_C=64,
            num_warps=8,
        )
        return output

    return sigmoid_conv2d


def build(context):
    return _build_sigmoid_conv2d(context)
