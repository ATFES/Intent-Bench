from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.fn
def _conv_value(
    input: I.In[I.f32, ("B", "C", "H", "W")],
    weight: I.In[I.f32, ("N", "CG", "KH", "KW")],
    output_h: I.i64,
    output_w: I.i64,
    stride_h: I.i64,
    stride_w: I.i64,
    padding_h: I.i64,
    padding_w: I.i64,
    dilation_h: I.i64,
    dilation_w: I.i64,
):
    output_h = I.cast(output_h, I.index)
    output_w = I.cast(output_w, I.index)
    stride_h = I.cast(stride_h, I.index)
    stride_w = I.cast(stride_w, I.index)
    padding_h = I.cast(padding_h, I.index)
    padding_w = I.cast(padding_w, I.index)
    dilation_h = I.cast(dilation_h, I.index)
    dilation_w = I.cast(dilation_w, I.index)
    batch, channels, input_h, input_w = input.shape
    outputs, channels_per_group, kernel_h, kernel_w = weight.shape
    groups = channels // channels_per_group
    outputs_per_group = outputs // groups
    reduction = channels_per_group * kernel_h * kernel_w
    batch_coordinates = I.reshape(
        I.indices(I.domain(0, batch)),
        (batch, 1, 1, 1, 1, 1, 1),
    )
    row_coordinates = I.reshape(
        I.indices(I.domain(0, output_h)),
        (1, output_h, 1, 1, 1, 1, 1),
    )
    column_coordinates = I.reshape(
        I.indices(I.domain(0, output_w)),
        (1, 1, output_w, 1, 1, 1, 1),
    )
    group_coordinates = I.reshape(
        I.indices(I.domain(0, groups)),
        (1, 1, 1, groups, 1, 1, 1),
    )
    channel_coordinates = I.reshape(
        I.indices(I.domain(0, channels_per_group)),
        (1, 1, 1, 1, channels_per_group, 1, 1),
    )
    kernel_row_coordinates = I.reshape(
        I.indices(I.domain(0, kernel_h)),
        (1, 1, 1, 1, 1, kernel_h, 1),
    )
    kernel_column_coordinates = I.reshape(
        I.indices(I.domain(0, kernel_w)),
        (1, 1, 1, 1, 1, 1, kernel_w),
    )
    source_channel = group_coordinates * channels_per_group + channel_coordinates
    source_row = (
        row_coordinates * stride_h
        - padding_h
        + kernel_row_coordinates * dilation_h
    )
    source_column = (
        column_coordinates * stride_w
        - padding_w
        + kernel_column_coordinates * dilation_w
    )
    valid = (
        (source_row >= 0)
        & (source_row < input_h)
        & (source_column >= 0)
        & (source_column < input_w)
    )
    patches = I.gather(
        input,
        (
            batch_coordinates,
            source_channel,
            source_row,
            source_column,
        ),
        valid=valid,
        fill=I.cast(0.0, I.f32),
    )

    grouped_weight = I.reshape(
        weight,
        (groups, outputs_per_group, channels_per_group, kernel_h, kernel_w),
    )
    weight_for_vecmat = I.reshape(
        I.transpose(
            grouped_weight,
            (0, 2, 3, 4, 1),
        ),
        (groups, reduction, outputs_per_group),
    )
    return I.vecmat(
        patches,
        weight_for_vecmat,
        acc_dtype=I.f32,
    )


@intent.kernel
def conv2d_with_bias(
    input: I.In[I.f32, ("B", "C", "H", "W")],
    weight: I.In[I.f32, ("N", "CG", "KH", "KW")],
    output: I.Out[I.f32, ("B", "OH", "OW", "N")],
    bias: I.In[I.f32, ("N",)],
    stride_h: I.i64,
    stride_w: I.i64,
    padding_h: I.i64,
    padding_w: I.i64,
    dilation_h: I.i64,
    dilation_w: I.i64,
):
    batch, _, _, _ = input.shape
    outputs, _, _, _ = weight.shape
    output_h, output_w = output.shape[1], output.shape[2]
    value = _conv_value(
        input,
        weight,
        output_h,
        output_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )
    outputs_value = I.reshape(value, (batch, output_h, output_w, outputs))
    biased = outputs_value + I.reshape(bias, (1, 1, 1, outputs))
    output[
        I.domain(0, batch),
        I.domain(0, output_h),
        I.domain(0, output_w),
        I.domain(0, outputs),
    ] = biased


@intent.kernel
def conv2d_without_bias(
    input: I.In[I.f32, ("B", "C", "H", "W")],
    weight: I.In[I.f32, ("N", "CG", "KH", "KW")],
    output: I.Out[I.f32, ("B", "OH", "OW", "N")],
    stride_h: I.i64,
    stride_w: I.i64,
    padding_h: I.i64,
    padding_w: I.i64,
    dilation_h: I.i64,
    dilation_w: I.i64,
):
    batch, _, _, _ = input.shape
    outputs, _, _, _ = weight.shape
    output_h, output_w = output.shape[1], output.shape[2]
    value = _conv_value(
        input,
        weight,
        output_h,
        output_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        dilation_h,
        dilation_w,
    )
    output[
        I.domain(0, batch),
        I.domain(0, output_h),
        I.domain(0, output_w),
        I.domain(0, outputs),
    ] = I.reshape(value, (batch, output_h, output_w, outputs))


def _pair(value, name: str):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must contain two values")
        return int(value[0]), int(value[1])
    repeated = int(value)
    return repeated, repeated


def _padding_pair(value, kernel, stride, input_size, name: str):
    if value == "same":
        output_size = (input_size + stride - 1) // stride
        total = max((output_size - 1) * stride + kernel - input_size, 0)
        return total // 2, total - total // 2
    if value == "valid":
        return 0, 0
    return _pair(value, name)


def build(context):
    with_bias = context.compile("conv2d_bias", conv2d_with_bias)
    without_bias = context.compile("conv2d_no_bias", conv2d_without_bias)

    def conv2d(
            input,
            weight,
            output_h,
            output_w,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        stride_h, stride_w = _pair(stride, "stride")
        dilation_h, dilation_w = _pair(dilation, "dilation")
        top_padding, bottom_padding = _padding_pair(
            padding,
            (weight.shape[2] - 1) * dilation_h + 1,
            stride_h,
            input.shape[2],
            "padding",
        )
        left_padding, right_padding = _padding_pair(
            padding,
            (weight.shape[3] - 1) * dilation_w + 1,
            stride_w,
            input.shape[3],
            "padding",
        )
        output_h = (
            input.shape[2]
            + top_padding
            + bottom_padding
            - (weight.shape[2] - 1) * dilation_h
            - 1
        ) // stride_h + 1
        output_w = (
            input.shape[3]
            + left_padding
            + right_padding
            - (weight.shape[3] - 1) * dilation_w
            - 1
        ) // stride_w + 1
        output = torch.empty(
            (input.shape[0], weight.shape[0], output_h, output_w),
            dtype=input.dtype,
            device=input.device,
        )
        if bias is None:
            without_bias(
                input,
                weight,
                output_h,
                output_w,
                output,
                stride_h,
                stride_w,
                top_padding,
                left_padding,
                dilation_h,
                dilation_w,
            )
        else:
            with_bias(
                input,
                weight,
                output,
                bias,
                stride_h,
                stride_w,
                top_padding,
                left_padding,
                dilation_h,
                dilation_w,
            )
        return output

    return conv2d
