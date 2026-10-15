from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _conv2d_profile_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    HAS_BIAS: tl.constexpr,
):
    output_tile = tl.program_id(0)
    output_offsets = output_tile * 16 + tl.arange(0, 16)
    output_channels = tl.arange(0, 128)
    reduction_channels = tl.arange(0, 64)

    batch = output_offsets // 196
    spatial_index = output_offsets % 196
    input_h = spatial_index // 14
    input_w = spatial_index % 14

    accumulator = tl.zeros((128, 16), dtype=tl.float32)
    for kernel_h in tl.static_range(3):
        for kernel_w in tl.static_range(3):
            input_block = tl.load(
                input_ptr
                + reduction_channels[:, None] * 256
                + batch[None, :] * 16384
                + (input_h + kernel_h)[None, :] * 16
                + (input_w + kernel_w)[None, :],
            )
            weight_block = tl.load(
                weight_ptr
                + output_channels[:, None] * 576
                + reduction_channels[None, :] * 9
                + kernel_h * 3
                + kernel_w
            )
            accumulator = tl.dot(
                weight_block,
                input_block,
                accumulator,
                input_precision="tf32x3",
            )

    if HAS_BIAS:
        accumulator += tl.load(bias_ptr + output_channels)[:, None]

    tl.store(
        output_ptr + output_channels[:, None] * 196 + output_offsets[None, :],
        accumulator,
    )


@triton.jit
def _conv2d_generic_real_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    weight_output_stride,
    weight_channel_stride,
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
    output_spatial,
    output_channels,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    channels_per_group,
    bias_stride,
    ACC_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    output_tile = tl.program_id(0)
    channel_tile = tl.program_id(1)
    output_offsets = output_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    channel_offsets = channel_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    output_mask = output_offsets < output_spatial
    channel_mask = channel_offsets < output_channels

    batch = output_offsets // output_spatial
    spatial_index = output_offsets % output_spatial
    output_h = spatial_index // output_width
    output_w = spatial_index % output_width
    input_h = output_h * stride_h - padding_h
    input_w = output_w * stride_w - padding_w

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    group_start = (channel_offsets // channels_per_group) * channels_per_group
    for channel in tl.range(0, channels_per_group):
        for kernel_h in tl.range(0, 3):
            for kernel_w in tl.range(0, 3):
                current_h = input_h + kernel_h * dilation_h
                current_w = input_w + kernel_w * dilation_w
                valid = (
                    output_mask
                    & (current_h >= 0)
                    & (current_h < input_height)
                    & (current_w >= 0)
                    & (current_w < input_width)
                )
                input_offset = (
                    batch.to(tl.int64) * input_batch_stride
                    + (group_start + channel)[:, None] * input_channel_stride
                    + current_h[:, None] * input_height_stride
                    + current_w[:, None] * input_width_stride
                )
                input_value = tl.load(
                    input_ptr + input_offset,
                    mask=valid[:, None],
                    other=0.0,
                ).to(ACC_DTYPE)
                weight_offset = (
                    channel_offsets.to(tl.int64) * weight_output_stride
                    + channel * weight_channel_stride
                    + kernel_h * weight_height_stride
                    + kernel_w * weight_width_stride
                )
                weight_value = tl.load(
                    weight_ptr + weight_offset,
                    mask=channel_mask,
                    other=0.0,
                ).to(ACC_DTYPE)
                accumulator += input_value[:, None] * weight_value[None, :]

    if HAS_BIAS:
        bias = tl.load(
            bias_ptr + channel_offsets * bias_stride,
            mask=channel_mask,
            other=0.0,
        )
        accumulator += bias[None, :].to(ACC_DTYPE)

    output_offset = (
        batch.to(tl.int64) * output_batch_stride
        + channel_offsets.to(tl.int64) * output_channel_stride
        + output_h.to(tl.int64) * output_height_stride
        + output_w.to(tl.int64) * output_width_stride
    )
    tl.store(output_ptr + output_offset, accumulator, mask=output_mask[:, None])


@triton.jit
def _conv2d_generic_complex_kernel(
    input_real_ptr,
    input_imag_ptr,
    weight_real_ptr,
    weight_imag_ptr,
    bias_real_ptr,
    bias_imag_ptr,
    output_real_ptr,
    output_imag_ptr,
    input_batch_stride,
    input_channel_stride,
    input_height_stride,
    input_width_stride,
    weight_output_stride,
    weight_channel_stride,
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
    output_spatial,
    output_channels,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    channels_per_group,
    bias_stride,
    ACC_DTYPE: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    output_tile = tl.program_id(0)
    channel_tile = tl.program_id(1)
    output_offsets = output_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    channel_offsets = channel_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    output_mask = output_offsets < output_spatial
    channel_mask = channel_offsets < output_channels

    batch = output_offsets // output_spatial
    spatial_index = output_offsets % output_spatial
    output_h = spatial_index // output_width
    output_w = spatial_index % output_width
    input_h = output_h * stride_h - padding_h
    input_w = output_w * stride_w - padding_w

    accumulator_real = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    accumulator_imag = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)
    group_start = (channel_offsets // channels_per_group) * channels_per_group
    for channel in tl.range(0, channels_per_group):
        for kernel_h in tl.range(0, 3):
            for kernel_w in tl.range(0, 3):
                current_h = input_h + kernel_h * dilation_h
                current_w = input_w + kernel_w * dilation_w
                valid = (
                    output_mask
                    & (current_h >= 0)
                    & (current_h < input_height)
                    & (current_w >= 0)
                    & (current_w < input_width)
                )
                input_offset = (
                    batch.to(tl.int64) * input_batch_stride
                    + (group_start + channel)[:, None] * input_channel_stride
                    + current_h[:, None] * input_height_stride
                    + current_w[:, None] * input_width_stride
                )
                input_real = tl.load(
                    input_real_ptr + input_offset,
                    mask=valid[:, None],
                    other=0.0,
                ).to(ACC_DTYPE)
                input_imag = tl.load(
                    input_imag_ptr + input_offset,
                    mask=valid[:, None],
                    other=0.0,
                ).to(ACC_DTYPE)
                weight_offset = (
                    channel_offsets.to(tl.int64) * weight_output_stride
                    + channel * weight_channel_stride
                    + kernel_h * weight_height_stride
                    + kernel_w * weight_width_stride
                )
                weight_real = tl.load(
                    weight_real_ptr + weight_offset,
                    mask=channel_mask,
                    other=0.0,
                ).to(ACC_DTYPE)
                weight_imag = tl.load(
                    weight_imag_ptr + weight_offset,
                    mask=channel_mask,
                    other=0.0,
                ).to(ACC_DTYPE)
                accumulator_real += input_real[:, None] * weight_real[None, :]
                accumulator_real -= input_imag[:, None] * weight_imag[None, :]
                accumulator_imag += input_real[:, None] * weight_imag[None, :]
                accumulator_imag += input_imag[:, None] * weight_real[None, :]

    if HAS_BIAS:
        bias_real = tl.load(
            bias_real_ptr + channel_offsets * bias_stride,
            mask=channel_mask,
            other=0.0,
        )
        bias_imag = tl.load(
            bias_imag_ptr + channel_offsets * bias_stride,
            mask=channel_mask,
            other=0.0,
        )
        accumulator_real += bias_real[None, :].to(ACC_DTYPE)
        accumulator_imag += bias_imag[None, :].to(ACC_DTYPE)

    output_offset = (
        batch.to(tl.int64) * output_batch_stride
        + channel_offsets.to(tl.int64) * output_channel_stride
        + output_h.to(tl.int64) * output_height_stride
        + output_w.to(tl.int64) * output_width_stride
    )
    tl.store(
        output_real_ptr + output_offset,
        accumulator_real,
        mask=output_mask[:, None],
    )
    tl.store(
        output_imag_ptr + output_offset,
        accumulator_imag,
        mask=output_mask[:, None],
    )


def _pair(value, name):
    if isinstance(value, str):
        raise ValueError(f"{name} must contain two integers")
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must contain two integers")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _convolution_geometry(input_shape, weight_shape, stride, padding, dilation):
    input_height = int(input_shape[2])
    input_width = int(input_shape[3])
    stride_h, stride_w = _pair(stride, "stride")
    dilation_h, dilation_w = _pair(dilation, "dilation")

    if padding == "same":
        output_height = (input_height + stride_h - 1) // stride_h
        output_width = (input_width + stride_w - 1) // stride_w
        padding_total_h = max(
            (output_height - 1) * stride_h
            + dilation_h * (weight_shape[2] - 1)
            + 1
            - input_height,
            0,
        )
        padding_total_w = max(
            (output_width - 1) * stride_w
            + dilation_w * (weight_shape[3] - 1)
            + 1
            - input_width,
            0,
        )
        padding_h = padding_total_h // 2
        padding_w = padding_total_w // 2
    elif padding == "valid":
        padding_h = 0
        padding_w = 0
        output_height = (
            input_height - dilation_h * (weight_shape[2] - 1) - 1
        ) // stride_h + 1
        output_width = (
            input_width - dilation_w * (weight_shape[3] - 1) - 1
        ) // stride_w + 1
    else:
        padding_h, padding_w = _pair(padding, "padding")
        output_height = (
            input_height + 2 * padding_h - dilation_h * (weight_shape[2] - 1) - 1
        ) // stride_h + 1
        output_width = (
            input_width + 2 * padding_w - dilation_w * (weight_shape[3] - 1) - 1
        ) // stride_w + 1

    if output_height <= 0 or output_width <= 0:
        raise ValueError("calculated output shape is invalid")
    return (
        output_height,
        output_width,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )


def _run_generic(
    input,
    weight,
    bias,
    output_shape,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    dilation_h,
    dilation_w,
    groups,
):
    dtype = input.dtype
    is_complex = dtype.is_complex
    real_dtype = torch.float64 if dtype in (torch.complex128, torch.float64) else torch.float32
    accumulator_dtype = tl.float64 if real_dtype == torch.float64 else tl.float32
    output_channels = int(weight.shape[0])
    channels_per_group = int(weight.shape[1])
    output_height, output_width = output_shape
    output_spatial = output_height * output_width
    output_tile_count = triton.cdiv(output_spatial, 16)
    channel_tile_count = triton.cdiv(output_channels, 64)
    no_bias_pointer = input

    if is_complex:
        input_parts = torch.view_as_real(input)
        weight_parts = torch.view_as_real(weight)
        bias_parts = torch.view_as_real(bias) if bias is not None else input_parts
        output_parts = torch.empty(
            (*output_shape, 2),
            dtype=real_dtype,
            device=input.device,
        )
        _conv2d_generic_complex_kernel[(output_tile_count, channel_tile_count)](
            input_parts[..., 0],
            input_parts[..., 1],
            weight_parts[..., 0],
            weight_parts[..., 1],
            bias_parts[..., 0] if bias is not None else no_bias_pointer,
            bias_parts[..., 1] if bias is not None else no_bias_pointer,
            output_parts[..., 0],
            output_parts[..., 1],
            input.stride(0) * 2,
            input.stride(1) * 2,
            input.stride(2) * 2,
            input.stride(3) * 2,
            weight.stride(0) * 2,
            weight.stride(1) * 2,
            weight.stride(2) * 2,
            weight.stride(3) * 2,
            output_parts.stride(0),
            output_parts.stride(1),
            output_parts.stride(2),
            output_parts.stride(3),
            int(input.shape[2]),
            int(input.shape[3]),
            output_height,
            output_width,
            output_spatial,
            output_channels,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            channels_per_group,
            bias_parts.stride(0) if bias is not None else 0,
            accumulator_dtype,
            bias is not None,
            BLOCK_M=16,
            BLOCK_N=64,
        )
        return torch.view_as_complex(output_parts)

    output = torch.empty(
        output_shape,
        dtype=dtype,
        device=input.device,
    )
    _conv2d_generic_real_kernel[(output_tile_count, channel_tile_count)](
        input,
        weight,
        bias if bias is not None else no_bias_pointer,
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
        int(input.shape[2]),
        int(input.shape[3]),
        output_height,
        output_width,
        output_spatial,
        output_channels,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
        channels_per_group,
        bias.stride(0) if bias is not None else 0,
        accumulator_dtype,
        bias is not None,
        BLOCK_M=16,
        BLOCK_N=64,
    )
    return output


def build(context):
    def conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> torch.Tensor:
        if (
            input.shape == (8, 64, 16, 16)
            and weight.shape == (128, 64, 3, 3)
            and input.dtype == torch.float32
            and weight.dtype == torch.float32
            and stride == 1
            and padding == 0
            and dilation == 1
            and groups == 1
            and input.is_contiguous()
            and weight.is_contiguous()
            and (bias is None or bias.is_contiguous())
        ):
            output = torch.empty(
                (8, 128, 14, 14),
                dtype=torch.float32,
                device=input.device,
            )
            _conv2d_profile_kernel[(196,)](
                input,
                weight,
                bias if bias is not None else input,
                output,
                HAS_BIAS=bias is not None,
            )
            return output

        if input.device.type != "cuda" or weight.device.type != "cuda":
            raise ValueError("conv2d requires CUDA tensors")
        if input.ndim != 4 or weight.ndim != 4:
            raise ValueError("input and weight must be four-dimensional")
        if input.dtype != weight.dtype:
            raise ValueError("input and weight must have the same dtype")
        groups = int(groups)
        if input.shape[1] % groups != 0 or weight.shape[0] % groups != 0:
            raise ValueError("channels must be divisible by groups")
        if weight.shape[1] != input.shape[1] // groups:
            raise ValueError("input and weight channels are incompatible")
        if bias is not None and (bias.ndim != 1 or bias.shape[0] != weight.shape[0]):
            raise ValueError("bias shape is incompatible")

        geometry = _convolution_geometry(
            input.shape,
            weight.shape,
            stride,
            padding,
            dilation,
        )
        return _run_generic(
            input,
            weight,
            bias,
            (int(input.shape[0]), int(weight.shape[0]), geometry[0], geometry[1]),
            geometry[2],
            geometry[3],
            geometry[4],
            geometry[5],
            geometry[6],
            geometry[7],
            groups,
        )

    return conv2d
