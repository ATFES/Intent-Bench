from __future__ import annotations

from typing import Callable

import intent
import intent.language as I
import torch


def _pair(value: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("expected an integer or a pair of integers")
    return int(value[0]), int(value[1])


@intent.kernel
def _pad_input(
    input: I.In[I.f32, ("B", "CIN", "IH", "IW")],
    padded: I.Out[I.f32, ("B", "CIN", "EXT_H", "EXT_W")],
    PAD_H: I.Constexpr[int],
    PAD_W: I.Constexpr[int],
    IH: I.Constexpr[int],
    IW: I.Constexpr[int],
) -> None:
    source_rows = I.indices(I.domain(0, EXT_H)) - I.cast(PAD_H, I.index)
    source_columns = I.indices(I.domain(0, EXT_W)) - I.cast(PAD_W, I.index)
    rows_valid = I.select(
        source_rows >= 0,
        source_rows < I.cast(IH, I.index),
        I.full(source_rows.shape, False, I.bool),
    )
    columns_valid = I.select(
        source_columns >= 0,
        source_columns < I.cast(IW, I.index),
        I.full(source_columns.shape, False, I.bool),
    )
    rows = I.gather(
        input,
        (slice(None), slice(None), source_rows, slice(None)),
        valid=rows_valid,
        fill=I.cast(0.0, I.f32),
    )
    padded_values = I.gather(
        rows,
        (slice(None), slice(None), slice(None), source_columns),
        valid=columns_valid,
        fill=I.cast(0.0, I.f32),
    )
    padded[:, :, :, :] = padded_values


@intent.kernel
def _conv_relu_no_bias(
    padded: I.In[I.f32, ("B", "CPG", "EXT_H", "EXT_W")],
    weight: I.In[I.f32, ("OG", "CPG", "KH", "KW")],
    output: I.Out[I.f32, ("B", "OG", "OH", "OW")],
    CPG: I.Constexpr[int],
    KH: I.Constexpr[int],
    KW: I.Constexpr[int],
    STRIDE_H: I.Constexpr[int],
    STRIDE_W: I.Constexpr[int],
    PAD_H: I.Constexpr[int],
    PAD_W: I.Constexpr[int],
    DILATION_H: I.Constexpr[int],
    DILATION_W: I.Constexpr[int],
    OH: I.Constexpr[int],
    OW: I.Constexpr[int],
) -> None:
    output_rows = I.indices(I.domain(0, OH))
    kernel_rows = I.indices(I.domain(0, KH))
    output_columns = I.indices(I.domain(0, OW))
    kernel_columns = I.indices(I.domain(0, KW))
    output_rows_column = I.reshape(output_rows, (OH, 1))
    kernel_rows_row = I.reshape(kernel_rows, (1, KH))
    input_rows = (
        output_rows_column * I.cast(STRIDE_H, I.index)
        + kernel_rows_row * I.cast(DILATION_H, I.index)
    )
    output_columns_column = I.reshape(output_columns, (OW, 1))
    kernel_columns_row = I.reshape(kernel_columns, (1, KW))
    input_columns = (
        output_columns_column * I.cast(STRIDE_W, I.index)
        + kernel_columns_row * I.cast(DILATION_W, I.index)
    )
    row_coordinates = I.reshape(input_rows, (OH, KH, 1, 1))
    column_coordinates = I.reshape(input_columns, (1, 1, OW, KW))
    patches = I.gather(
        padded,
        (
            slice(None),
            slice(None),
            row_coordinates,
            column_coordinates,
        ),
    )
    convolution_layout = I.transpose(patches, (0, 2, 4, 1, 3, 5))
    convolution = I.contract(
        convolution_layout,
        weight,
        reduce=((3, 1), (4, 2), (5, 3)),
        acc_dtype=I.f32,
    )
    output_layout = I.transpose(convolution, (0, 3, 1, 2))
    output[:, :, :, :] = I.maximum(
        output_layout,
        I.full(output_layout.shape, 0.0, I.f32),
    )


@intent.kernel
def _conv_relu_bias(
    padded: I.In[I.f32, ("B", "CPG", "EXT_H", "EXT_W")],
    weight: I.In[I.f32, ("OG", "CPG", "KH", "KW")],
    bias: I.In[I.f32, ("OG",)],
    output: I.Out[I.f32, ("B", "OG", "OH", "OW")],
    CPG: I.Constexpr[int],
    KH: I.Constexpr[int],
    KW: I.Constexpr[int],
    STRIDE_H: I.Constexpr[int],
    STRIDE_W: I.Constexpr[int],
    PAD_H: I.Constexpr[int],
    PAD_W: I.Constexpr[int],
    DILATION_H: I.Constexpr[int],
    DILATION_W: I.Constexpr[int],
    OH: I.Constexpr[int],
    OW: I.Constexpr[int],
) -> None:
    output_rows = I.indices(I.domain(0, OH))
    kernel_rows = I.indices(I.domain(0, KH))
    output_columns = I.indices(I.domain(0, OW))
    kernel_columns = I.indices(I.domain(0, KW))
    output_rows_column = I.reshape(output_rows, (OH, 1))
    kernel_rows_row = I.reshape(kernel_rows, (1, KH))
    input_rows = (
        output_rows_column * I.cast(STRIDE_H, I.index)
        + kernel_rows_row * I.cast(DILATION_H, I.index)
    )
    output_columns_column = I.reshape(output_columns, (OW, 1))
    kernel_columns_row = I.reshape(kernel_columns, (1, KW))
    input_columns = (
        output_columns_column * I.cast(STRIDE_W, I.index)
        + kernel_columns_row * I.cast(DILATION_W, I.index)
    )
    row_coordinates = I.reshape(input_rows, (OH, KH, 1, 1))
    column_coordinates = I.reshape(input_columns, (1, 1, OW, KW))
    patches = I.gather(
        padded,
        (
            slice(None),
            slice(None),
            row_coordinates,
            column_coordinates,
        ),
    )
    convolution_layout = I.transpose(patches, (0, 2, 4, 1, 3, 5))
    convolution = I.contract(
        convolution_layout,
        weight,
        reduce=((3, 1), (4, 2), (5, 3)),
        acc_dtype=I.f32,
    )
    bias_broadcast = I.reshape(bias, (1, OG, 1, 1))
    convolution = convolution + bias_broadcast
    output_layout = I.transpose(convolution, (0, 3, 1, 2))
    output[:, :, :, :] = I.maximum(
        output_layout,
        I.full(output_layout.shape, 0.0, I.f32),
    )


def build(context) -> Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        int | tuple[int, int],
        int | tuple[int, int] | str,
        int | tuple[int, int],
        int,
        bool,
    ],
    torch.Tensor,
]:
    input_shape = tuple(input_shape for input_shape in (1, 3, 256, 256))
    batch_size, input_channels, input_height, input_width = input_shape
    output_channels, channels_per_group, kernel_height, kernel_width = (64, 3, 3, 3)
    stride_h, stride_w = 1, 1
    padding_h, padding_w = 1, 1
    dilation_h, dilation_w = 1, 1
    groups = 1
    extended_height = input_height + 2 * padding_h
    extended_width = input_width + 2 * padding_w
    output_height = (
        input_height
        + 2 * padding_h
        - dilation_h * (kernel_height - 1)
        - 1
    ) // stride_h + 1
    output_width = (
        input_width
        + 2 * padding_w
        - dilation_w * (kernel_width - 1)
        - 1
    ) // stride_w + 1

    no_bias = context.compile(
        "relu_conv2d_no_bias",
        _conv_relu_no_bias,
        constexprs={
            "B": batch_size,
            "CPG": channels_per_group,
            "EXT_H": extended_height,
            "EXT_W": extended_width,
            "OG": output_channels // groups,
            "KH": kernel_height,
            "KW": kernel_width,
            "STRIDE_H": stride_h,
            "STRIDE_W": stride_w,
            "PAD_H": padding_h,
            "PAD_W": padding_w,
            "DILATION_H": dilation_h,
            "DILATION_W": dilation_w,
            "OH": output_height,
            "OW": output_width,
        },
    )
    with_bias = context.compile(
        "relu_conv2d_with_bias",
        _conv_relu_bias,
        constexprs={
            "B": batch_size,
            "CPG": channels_per_group,
            "EXT_H": extended_height,
            "EXT_W": extended_width,
            "OG": output_channels // groups,
            "KH": kernel_height,
            "KW": kernel_width,
            "STRIDE_H": stride_h,
            "STRIDE_W": stride_w,
            "PAD_H": padding_h,
            "PAD_W": padding_w,
            "DILATION_H": dilation_h,
            "DILATION_W": dilation_w,
            "OH": output_height,
            "OW": output_width,
        },
    )
    pad_artifact = context.compile(
        "relu_conv2d_pad",
        _pad_input,
        constexprs={
            "B": batch_size,
            "CIN": input_channels,
            "IH": input_height,
            "IW": input_width,
            "EXT_H": extended_height,
            "EXT_W": extended_width,
            "PAD_H": padding_h,
            "PAD_W": padding_w,
        },
    )

    def relu_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        inplace=False,
    ):
        if input.dtype != torch.float32 or weight.dtype != torch.float32:
            raise TypeError("relu_conv2d requires float32 tensors")
        if input.ndim != 4 or weight.ndim != 4:
            raise ValueError("input and weight must be four-dimensional")
        if tuple(input.shape[1:]) != (3, 256, 256):
            raise ValueError("unsupported input shape for compiled profile")
        if tuple(weight.shape) != (64, 3, 3, 3):
            raise ValueError("unsupported weight shape for compiled profile")
        stride_pair = _pair(stride)
        dilation_pair = _pair(dilation)
        if stride_pair != (1, 1) or dilation_pair != (1, 1) or groups != 1:
            raise ValueError("unsupported convolution configuration for compiled profile")
        if padding == "same":
            padding_pair = (1, 1)
        elif padding == "valid":
            padding_pair = (0, 0)
        else:
            padding_pair = _pair(padding)
        if padding_pair != (1, 1):
            raise ValueError("unsupported padding for compiled profile")

        batch_size, input_channels, input_height, input_width = input.shape
        output_channels, channels_per_group, kernel_height, kernel_width = weight.shape
        extended_height = input_height + 2 * padding_pair[0]
        extended_width = input_width + 2 * padding_pair[1]
        profile_output_height = (
            input_height
            + 2 * padding_pair[0]
            - dilation_pair[0] * (kernel_height - 1)
            - 1
        ) // stride_pair[0] + 1
        profile_output_width = (
            input_width
            + 2 * padding_pair[1]
            - dilation_pair[1] * (kernel_width - 1)
            - 1
        ) // stride_pair[1] + 1
        padded = torch.empty(
            (batch_size, input_channels, extended_height, extended_width),
            device=input.device,
            dtype=input.dtype,
        )
        output = torch.empty(
            (
                batch_size,
                output_channels,
                profile_output_height,
                profile_output_width,
            ),
            device=input.device,
            dtype=input.dtype,
        )
        pad_artifact(
            input,
            padded,
        )
        input_group_size = input_channels // groups
        output_group_size = output_channels // groups
        for group in range(groups):
            group_input = padded[
                :,
                group * input_group_size : (group + 1) * input_group_size,
                :,
                :,
            ]
            group_weight = weight[
                group * output_group_size : (group + 1) * output_group_size
            ]
            group_output = output[
                :,
                group * output_group_size : (group + 1) * output_group_size,
                :,
                :,
            ]
            if bias is None:
                no_bias(group_input, group_weight, group_output)
            else:
                group_bias = bias[
                    group * output_group_size : (group + 1) * output_group_size
                ]
                with_bias(group_input, group_weight, group_bias, group_output)
        return output

    return relu_conv2d
