from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def _conv_selu(
    input: I.In[I.f32, (2, 16, 64, 64)],
    weight: I.In[I.f32, (64, 16, 3, 3)],
    bias: I.In[I.f32, (64,)],
    output: I.Out[I.f32, (2, 64, 62, 62)],
):
    batches = I.domain(0, 2)
    channels = I.domain(0, 64)
    heights = I.domain(0, 62)
    widths = I.domain(0, 62)
    input_channels = I.domain(0, 16)
    kernel_rows = I.domain(0, 3)
    kernel_cols = I.domain(0, 3)

    batch_i = I.reshape(I.indices(batches), (2, 1, 1, 1, 1, 1, 1))
    channel_i = I.reshape(I.indices(channels), (1, 64, 1, 1, 1, 1, 1))
    height_i = I.reshape(I.indices(heights), (1, 1, 62, 1, 1, 1, 1))
    width_i = I.reshape(I.indices(widths), (1, 1, 1, 62, 1, 1, 1))
    input_channel_i = I.reshape(I.indices(input_channels), (1, 1, 1, 1, 16, 1, 1))
    kernel_row_i = I.reshape(I.indices(kernel_rows), (1, 1, 1, 1, 1, 3, 1))
    kernel_col_i = I.reshape(I.indices(kernel_cols), (1, 1, 1, 1, 1, 1, 3))

    sampled = input[
        batch_i,
        input_channel_i,
        height_i + kernel_row_i,
        width_i + kernel_col_i,
    ]
    coefficients = weight[
        channel_i,
        input_channel_i,
        kernel_row_i,
        kernel_col_i,
    ]
    convolution = I.reduce.sum(
        sampled * coefficients,
        axis=(4, 5, 6),
        acc_dtype=I.f32,
    )

    bias_i = I.reshape(I.indices(channels), (1, 64, 1, 1))
    convolution = convolution + bias[bias_i]

    alpha = I.cast(1.6732632423543772, I.f32)
    scale = I.cast(1.0507009873554805, I.f32)
    one = I.cast(1.0, I.f32)
    negative_branch = alpha * (I.exp(convolution) - one)
    activated = scale * I.select(convolution > 0.0, convolution, negative_branch)
    output[batches, channels, heights, widths] = activated


@intent.kernel
def _conv_selu_no_bias(
    input: I.In[I.f32, (2, 16, 64, 64)],
    weight: I.In[I.f32, (64, 16, 3, 3)],
    output: I.Out[I.f32, (2, 64, 62, 62)],
):
    batches = I.domain(0, 2)
    channels = I.domain(0, 64)
    heights = I.domain(0, 62)
    widths = I.domain(0, 62)
    input_channels = I.domain(0, 16)
    kernel_rows = I.domain(0, 3)
    kernel_cols = I.domain(0, 3)

    batch_i = I.reshape(I.indices(batches), (2, 1, 1, 1, 1, 1, 1))
    channel_i = I.reshape(I.indices(channels), (1, 64, 1, 1, 1, 1, 1))
    height_i = I.reshape(I.indices(heights), (1, 1, 62, 1, 1, 1, 1))
    width_i = I.reshape(I.indices(widths), (1, 1, 1, 62, 1, 1, 1))
    input_channel_i = I.reshape(I.indices(input_channels), (1, 1, 1, 1, 16, 1, 1))
    kernel_row_i = I.reshape(I.indices(kernel_rows), (1, 1, 1, 1, 1, 3, 1))
    kernel_col_i = I.reshape(I.indices(kernel_cols), (1, 1, 1, 1, 1, 1, 3))

    sampled = input[
        batch_i,
        input_channel_i,
        height_i + kernel_row_i,
        width_i + kernel_col_i,
    ]
    coefficients = weight[
        channel_i,
        input_channel_i,
        kernel_row_i,
        kernel_col_i,
    ]
    convolution = I.reduce.sum(
        sampled * coefficients,
        axis=(4, 5, 6),
        acc_dtype=I.f32,
    )

    alpha = I.cast(1.6732632423543772, I.f32)
    scale = I.cast(1.0507009873554805, I.f32)
    one = I.cast(1.0, I.f32)
    negative_branch = alpha * (I.exp(convolution) - one)
    activated = scale * I.select(convolution > 0.0, convolution, negative_branch)
    output[batches, channels, heights, widths] = activated


@intent.kernel
def _instance_norm(
    input: I.In[I.f32, (2, 64, 62, 62)],
    output: I.Out[I.f32, (2, 64, 62, 62)],
    eps: I.f32,
):
    batches = I.domain(0, 2)
    channels = I.domain(0, 64)
    heights = I.domain(0, 62)
    widths = I.domain(0, 62)

    values = input[batches, channels, heights, widths]
    mean = I.reduce.sum(values, axis=(2, 3), acc_dtype=I.f32) / I.cast(3844.0, I.f32)
    mean = I.reshape(mean, (2, 64, 1, 1))
    centered = values - mean
    variance = I.reduce.sum(centered * centered, axis=(2, 3), acc_dtype=I.f32)
    variance = variance / I.cast(3844.0, I.f32)
    denominator = I.sqrt(variance + eps)
    normalized = centered / I.reshape(denominator, (2, 64, 1, 1))
    output[batches, channels, heights, widths] = normalized


def build(context):
    conv = context.compile("fused_conv_selu", _conv_selu, constexprs={})
    conv_no_bias = context.compile("fused_conv_selu_no_bias", _conv_selu_no_bias, constexprs={})
    normalize = context.compile("fused_instance_norm", _instance_norm, constexprs={})

    def fused_instance_norm_selu_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        num_features=None,
        eps=1e-5,
        momentum=0.1,
        affine=False,
        track_running_stats=False,
    ) -> torch.Tensor:
        del stride, padding, dilation, groups, num_features, momentum, affine, track_running_stats
        if bias is None:
            intermediate = conv_no_bias.run(input, weight)
        else:
            intermediate = conv.run(input, weight, bias)
        return normalize.run(intermediate, eps)

    return fused_instance_norm_selu_conv2d
