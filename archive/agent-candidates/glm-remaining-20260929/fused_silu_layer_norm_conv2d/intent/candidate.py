from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _make_patches(
    x: I.In[I.f32, ("B", "C", "H", "W")],
    conv_weight: I.In[I.f32, ("C", "R", "S")],
    patches: I.Out[I.f32, ("B", "K", "M")],
    conv_stride: I.i64,
    conv_padding: I.i64,
    conv_dilation: I.i64,
    DEFAULT_GEOMETRY: I.Constexpr[bool],
) -> None:
    batch_domain = I.domain(0, x.shape[0])
    input_channel_domain = I.domain(0, x.shape[1])
    kernel_row_domain = I.domain(0, conv_weight.shape[1])
    kernel_column_domain = I.domain(0, conv_weight.shape[2])
    position_domain = I.domain(0, patches.shape[2])
    input_height = x.shape[2]
    input_width = x.shape[3]
    kernel_height = conv_weight.shape[1]
    kernel_width = conv_weight.shape[2]
    stride = I.cast(conv_stride, I.index)
    padding = I.cast(conv_padding, I.index)
    dilation = I.cast(conv_dilation, I.index)
    output_width = (
        input_width + 2 * padding - dilation * (kernel_width - 1) - 1
    ) // stride + 1

    for batch in I.parallel(batch_domain):
        for input_channel in I.parallel(input_channel_domain):
            for kernel_row in I.parallel(kernel_row_domain):
                for kernel_column in I.parallel(kernel_column_domain):
                    patch_index = (
                        input_channel * kernel_height * kernel_width
                        + kernel_row * kernel_width
                        + kernel_column
                    )
                    for position in I.parallel(position_domain):
                        output_column = position % output_width
                        output_row = position // output_width
                        input_row = (
                            output_row * stride - padding + kernel_row * dilation
                        )
                        input_column = (
                            output_column * stride - padding + kernel_column * dilation
                        )
                        if DEFAULT_GEOMETRY:
                            patches[batch, patch_index, position] = x[
                                batch, input_channel, input_row, input_column
                            ]
                        else:
                            if (
                                input_row >= 0
                                and input_row < input_height
                                and input_column >= 0
                                and input_column < input_width
                            ):
                                patches[batch, patch_index, position] = x[
                                    batch, input_channel, input_row, input_column
                                ]
                            else:
                                patches[batch, patch_index, position] = 0.0


@intent.kernel
def _convolution_with_bias(
    conv_weight: I.In[I.f32, ("O", "K")],
    conv_bias: I.In[I.f32, ("O",)],
    patches: I.In[I.f32, ("B", "K", "M")],
    conv_output: I.Out[I.f32, ("B", "O", "M")],
) -> None:
    output_channels = conv_weight.shape[0]
    reduction_size = conv_weight.shape[1]
    batch_domain = I.domain(0, patches.shape[0])
    output_channels_domain = I.domain(0, output_channels)
    position_domain = I.domain(0, patches.shape[2])
    weight_matrix = I.reshape(
        conv_weight, (1, output_channels, reduction_size)
    )
    bias_column = I.reshape(
        conv_bias[output_channels_domain], (output_channels, 1)
    )

    products = I.matmul(
        weight_matrix,
        patches,
        acc_dtype=I.f32,
    )
    conv_output[batch_domain, output_channels_domain, position_domain] = (
        products + bias_column
    )


@intent.kernel
def _convolution_without_bias(
    conv_weight: I.In[I.f32, ("O", "K")],
    patches: I.In[I.f32, ("B", "K", "M")],
    conv_output: I.Out[I.f32, ("B", "O", "M")],
) -> None:
    output_channels = conv_weight.shape[0]
    reduction_size = conv_weight.shape[1]
    batch_domain = I.domain(0, patches.shape[0])
    output_channels_domain = I.domain(0, output_channels)
    position_domain = I.domain(0, patches.shape[2])
    weight_matrix = I.reshape(
        conv_weight, (1, output_channels, reduction_size)
    )

    products = I.matmul(
        weight_matrix,
        patches,
        acc_dtype=I.f32,
    )
    conv_output[batch_domain, output_channels_domain, position_domain] = products


@intent.kernel
def _normalization_statistics(
    conv_output: I.In[I.f32, ("B", "O", "M")],
    statistics: I.Out[I.f64, ("B", 2)],
    ln_eps: I.f64,
) -> None:
    batch_size = conv_output.shape[0]
    element_count = conv_output.shape[1] * conv_output.shape[2]
    element_count_f64 = I.cast(element_count, I.f64)

    for batch in I.parallel(I.domain(0, batch_size)):
        values = conv_output[batch, :, :]
        value_sum = I.reduce.sum(values, axis=(0, 1), acc_dtype=I.f64)
        squared_sum = I.reduce.sum(values * values, axis=(0, 1), acc_dtype=I.f64)
        mean = value_sum / element_count_f64
        variance = I.maximum(
            squared_sum / element_count_f64 - mean * mean,
            I.cast(0.0, I.f64),
        )
        reciprocal_standard_deviation = I.rsqrt(variance + ln_eps)
        statistics[batch, 0] = mean
        statistics[batch, 1] = reciprocal_standard_deviation


@intent.kernel
def _silu_layer_norm(
    conv_output: I.In[I.f32, ("B", "O", "M")],
    statistics: I.In[I.f64, ("B", 2)],
    result: I.Out[I.f32, ("B", "O", "M")],
) -> None:
    batch_size = conv_output.shape[0]
    output_channels = conv_output.shape[1]
    position_count = conv_output.shape[2]
    batch_domain = I.domain(0, batch_size)
    channel_domain = I.domain(0, output_channels)
    position_domain = I.domain(0, position_count)
    means = I.cast(
        I.reshape(statistics[batch_domain, 0], (batch_size, 1, 1)), I.f32
    )
    reciprocal_standard_deviations = I.cast(
        I.reshape(statistics[batch_domain, 1], (batch_size, 1, 1)), I.f32
    )

    values = conv_output[batch_domain, channel_domain, position_domain]
    normalized = (values - means) * reciprocal_standard_deviations
    result[batch_domain, channel_domain, position_domain] = (
        normalized * I.sigmoid(normalized)
    )


def build(context):
    patch_default = context.compile(
        "fused_conv_patches_default",
        _make_patches,
        constexprs={"DEFAULT_GEOMETRY": True},
    )
    patch_generic = context.compile(
        "fused_conv_patches_generic",
        _make_patches,
        constexprs={"DEFAULT_GEOMETRY": False},
    )
    convolution_with_bias = context.compile(
        "fused_convolution_with_bias", _convolution_with_bias
    )
    convolution_without_bias = context.compile(
        "fused_convolution_without_bias", _convolution_without_bias
    )
    statistics_kernel = context.compile(
        "fused_normalization_statistics", _normalization_statistics
    )
    silu_kernel = context.compile("fused_silu_layer_norm", _silu_layer_norm)

    def fused_silu_layer_norm_conv2d(
        x,
        weight,
        conv_weight,
        conv_bias=None,
        conv_stride=1,
        conv_padding=0,
        conv_dilation=1,
        conv_groups=1,
        ln_eps=1e-05,
    ):
        if conv_groups != 1:
            raise ValueError("this fixed-profile implementation requires conv_groups=1")
        if x.dtype != conv_weight.dtype:
            raise ValueError("x and conv_weight must have the same dtype")

        batch_size, in_channels, input_height, input_width = x.shape
        output_channels, group_channels, kernel_height, kernel_width = conv_weight.shape
        if group_channels != in_channels:
            raise ValueError("conv_weight input-channel extent must match x")

        output_height = (
            input_height + 2 * conv_padding - conv_dilation * (kernel_height - 1) - 1
        ) // conv_stride + 1
        output_width = (
            input_width + 2 * conv_padding - conv_dilation * (kernel_width - 1) - 1
        ) // conv_stride + 1
        reduction_size = in_channels * kernel_height * kernel_width
        position_count = output_height * output_width

        patches = torch.empty(
            (batch_size, reduction_size, position_count),
            device=x.device,
            dtype=x.dtype,
        )
        conv_output = torch.empty(
            (batch_size, output_channels, position_count),
            device=x.device,
            dtype=x.dtype,
        )
        statistics = torch.empty((batch_size, 2), device=x.device, dtype=torch.float64)
        result = torch.empty(
            (batch_size, output_channels, output_height, output_width),
            device=x.device,
            dtype=x.dtype,
        )

        patch_kernel = (
            patch_default
            if conv_stride == 1 and conv_padding == 0 and conv_dilation == 1
            else patch_generic
        )
        patch_kernel(
            x,
            conv_weight,
            patches,
            conv_stride,
            conv_padding,
            conv_dilation,
        )

        if conv_bias is None:
            convolution_without_bias(conv_weight, patches, conv_output)
        else:
            convolution_with_bias(conv_weight, conv_bias, patches, conv_output)

        statistics_kernel(conv_output, statistics, ln_eps)
        silu_kernel(conv_output, statistics, result)

        return result

    return fused_silu_layer_norm_conv2d
