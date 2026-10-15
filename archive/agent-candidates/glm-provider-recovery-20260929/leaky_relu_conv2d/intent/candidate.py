from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.fn
def _leaky_relu(value, negative_slope):
    zero = I.cast(0.0, I.f32)
    positive = I.maximum(value, zero)
    negative = I.minimum(value, zero)
    return positive + negative_slope * negative


@intent.kernel
def _leaky_relu_conv2d_with_bias(
    input: I.In[I.f32, ("B", "C", "H", "W")],
    weight: I.In[I.f32, ("O", "C", "KH", "KW")],
    bias: I.In[I.f32, ("O",)],
    output: I.Out[I.f32, ("B", "O", "OH", "OW")],
    stride_h: I.i64,
    stride_w: I.i64,
    padding_h: I.i64,
    padding_w: I.i64,
    dilation_h: I.i64,
    dilation_w: I.i64,
    negative_slope: I.f32,
):
    out_h = output.shape[2]
    out_w = output.shape[3]
    stride_h_index = I.cast(stride_h, I.index)
    stride_w_index = I.cast(stride_w, I.index)
    padding_h_index = I.cast(padding_h, I.index)
    padding_w_index = I.cast(padding_w, I.index)
    dilation_h_index = I.cast(dilation_h, I.index)
    dilation_w_index = I.cast(dilation_w, I.index)

    for batch in I.parallel(I.domain(0, input.shape[0])):
        accumulator = I.full(
            (weight.shape[0], out_h, out_w), fill=0.0, dtype=I.f32
        )
        for kernel_row in I.domain(0, weight.shape[2]):
            for kernel_column in I.domain(0, weight.shape[3]):
                row_start = (
                    kernel_row * dilation_h_index - padding_h_index
                )
                column_start = (
                    kernel_column * dilation_w_index - padding_w_index
                )
                row_stop = row_start + out_h
                column_stop = column_start + out_w
                patch = input[
                    batch,
                    :,
                    row_start:row_stop,
                    column_start:column_stop,
                ]
                flattened_patch = I.reshape(
                    patch, (input.shape[1], out_h * out_w)
                )
                tap_weight = weight[:, :, kernel_row, kernel_column]
                contribution = I.matmul(
                    tap_weight,
                    flattened_patch,
                    acc_dtype=I.f32,
                )
                contribution = I.reshape(
                    contribution, (weight.shape[0], out_h, out_w)
                )
                accumulator = accumulator + contribution

        shaped_bias = I.reshape(bias, (weight.shape[0], 1, 1))
        pre_activation = accumulator + shaped_bias
        output[batch, :, :, :] = _leaky_relu(pre_activation, negative_slope)


@intent.kernel
def _leaky_relu_conv2d_without_bias(
    input: I.In[I.f32, ("B", "C", "H", "W")],
    weight: I.In[I.f32, ("O", "C", "KH", "KW")],
    output: I.Out[I.f32, ("B", "O", "OH", "OW")],
    stride_h: I.i64,
    stride_w: I.i64,
    padding_h: I.i64,
    padding_w: I.i64,
    dilation_h: I.i64,
    dilation_w: I.i64,
    negative_slope: I.f32,
):
    out_h = output.shape[2]
    out_w = output.shape[3]
    stride_h_index = I.cast(stride_h, I.index)
    stride_w_index = I.cast(stride_w, I.index)
    padding_h_index = I.cast(padding_h, I.index)
    padding_w_index = I.cast(padding_w, I.index)
    dilation_h_index = I.cast(dilation_h, I.index)
    dilation_w_index = I.cast(dilation_w, I.index)

    for batch in I.parallel(I.domain(0, input.shape[0])):
        accumulator = I.full(
            (weight.shape[0], out_h, out_w), fill=0.0, dtype=I.f32
        )
        for kernel_row in I.domain(0, weight.shape[2]):
            for kernel_column in I.domain(0, weight.shape[3]):
                row_start = (
                    kernel_row * dilation_h_index - padding_h_index
                )
                column_start = (
                    kernel_column * dilation_w_index - padding_w_index
                )
                row_stop = row_start + out_h
                column_stop = column_start + out_w
                patch = input[
                    batch,
                    :,
                    row_start:row_stop,
                    column_start:column_stop,
                ]
                flattened_patch = I.reshape(
                    patch, (input.shape[1], out_h * out_w)
                )
                tap_weight = weight[:, :, kernel_row, kernel_column]
                contribution = I.matmul(
                    tap_weight,
                    flattened_patch,
                    acc_dtype=I.f32,
                )
                contribution = I.reshape(
                    contribution, (weight.shape[0], out_h, out_w)
                )
                accumulator = accumulator + contribution

        output[batch, :, :, :] = _leaky_relu(accumulator, negative_slope)


def build(context):
    with_bias = context.compile(
        "leaky_relu_conv2d_with_bias", _leaky_relu_conv2d_with_bias
    )
    without_bias = context.compile(
        "leaky_relu_conv2d_without_bias", _leaky_relu_conv2d_without_bias
    )

    def leaky_relu_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        negative_slope=0.01,
        inplace=False,
    ):
        if groups != 1:
            raise ValueError("this fixed-profile implementation requires groups == 1")

        stride_h = int(stride)
        stride_w = int(stride)
        padding_h = int(padding)
        padding_w = int(padding)
        dilation_h = int(dilation)
        dilation_w = int(dilation)
        slope = float(negative_slope)

        batch_size = input.shape[0]
        out_channels = weight.shape[0]
        out_height = (
            input.shape[2] + 2 * padding_h - dilation_h * (weight.shape[2] - 1) - 1
        ) // stride_h + 1
        out_width = (
            input.shape[3] + 2 * padding_w - dilation_w * (weight.shape[3] - 1) - 1
        ) // stride_w + 1
        output_shape = (batch_size, out_channels, out_height, out_width)

        if inplace and tuple(output_shape) == tuple(input.shape):
            output = input
        else:
            output = torch.empty(
                output_shape, dtype=input.dtype, device=input.device
            )

        if bias is None:
            without_bias(
                input,
                weight,
                output,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                slope,
            )
        else:
            with_bias(
                input,
                weight,
                bias,
                output,
                stride_h,
                stride_w,
                padding_h,
                padding_w,
                dilation_h,
                dilation_w,
                slope,
            )
        return output

    return leaky_relu_conv2d
