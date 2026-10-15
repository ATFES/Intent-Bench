from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def conv_bn_relu_no_bias(
    BATCH: I.Constexpr[int],
    INPUT_CHANNELS: I.Constexpr[int],
    INPUT_HEIGHT: I.Constexpr[int],
    INPUT_WIDTH: I.Constexpr[int],
    OUTPUT_CHANNELS: I.Constexpr[int],
    KERNEL_HEIGHT: I.Constexpr[int],
    KERNEL_WIDTH: I.Constexpr[int],
    OUTPUT_HEIGHT: I.Constexpr[int],
    OUTPUT_WIDTH: I.Constexpr[int],
    STRIDE: I.Constexpr[int],
    PADDING: I.Constexpr[int],
    DILATION: I.Constexpr[int],
    input: I.In[
        I.f32,
        (BATCH, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),
    ],
    weight: I.In[
        I.f32,
        (OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH),
    ],
    running_mean: I.In[I.f32, (OUTPUT_CHANNELS,)],
    running_var: I.In[I.f32, (OUTPUT_CHANNELS,)],
    bn_weight: I.In[I.f32, (OUTPUT_CHANNELS,)],
    bn_bias: I.In[I.f32, (OUTPUT_CHANNELS,)],
    eps: I.f32,
    output: I.Out[
        I.f32,
        (BATCH, OUTPUT_CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH),
    ],
) -> None:
    output_pixels = OUTPUT_HEIGHT * OUTPUT_WIDTH
    reduction_count = INPUT_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH
    kernel_size = KERNEL_HEIGHT * KERNEL_WIDTH

    output_rows = I.domain(0, output_pixels)
    output_columns = I.domain(0, OUTPUT_CHANNELS)
    reduction_columns = I.domain(0, reduction_count)
    output_channel_indices = I.reshape(
        I.indices(output_columns),
        (1, OUTPUT_CHANNELS),
    )

    row_indices = I.reshape(
        I.indices(output_rows),
        (output_pixels, 1),
    )
    reduction_indices = I.reshape(
        I.indices(reduction_columns),
        (1, reduction_count),
    )

    output_y = row_indices // OUTPUT_WIDTH
    output_x = row_indices % OUTPUT_WIDTH
    input_channels = reduction_indices // kernel_size
    kernel_remainder = reduction_indices % kernel_size
    kernel_y = kernel_remainder // KERNEL_WIDTH
    kernel_x = kernel_remainder % KERNEL_WIDTH
    input_y = (
        output_y * STRIDE - PADDING + kernel_y * DILATION
    )
    input_x = (
        output_x * STRIDE - PADDING + kernel_x * DILATION
    )

    input_patches = input[0, input_channels, input_y, input_x]
    weight_matrix = I.reshape(
        weight,
        (OUTPUT_CHANNELS, reduction_count),
    )
    weight_transposed = I.transpose(weight_matrix, (1, 0))
    convolution = I.matmul(
        input_patches,
        weight_transposed,
        acc_dtype=I.f32,
    )

    variance_epsilon = I.sqrt(running_var + eps)
    normalized = (convolution - running_mean) / variance_epsilon
    activated = I.maximum(
        normalized * bn_weight + bn_bias,
        I.cast(0.0, I.f32),
    )
    output[0, output_channel_indices, output_y, output_x] = activated


@intent.kernel
def conv_bn_relu_with_bias(
    BATCH: I.Constexpr[int],
    INPUT_CHANNELS: I.Constexpr[int],
    INPUT_HEIGHT: I.Constexpr[int],
    INPUT_WIDTH: I.Constexpr[int],
    OUTPUT_CHANNELS: I.Constexpr[int],
    KERNEL_HEIGHT: I.Constexpr[int],
    KERNEL_WIDTH: I.Constexpr[int],
    OUTPUT_HEIGHT: I.Constexpr[int],
    OUTPUT_WIDTH: I.Constexpr[int],
    STRIDE: I.Constexpr[int],
    PADDING: I.Constexpr[int],
    DILATION: I.Constexpr[int],
    input: I.In[
        I.f32,
        (BATCH, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH),
    ],
    weight: I.In[
        I.f32,
        (OUTPUT_CHANNELS, INPUT_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH),
    ],
    conv_bias: I.In[I.f32, (OUTPUT_CHANNELS,)],
    running_mean: I.In[I.f32, (OUTPUT_CHANNELS,)],
    running_var: I.In[I.f32, (OUTPUT_CHANNELS,)],
    bn_weight: I.In[I.f32, (OUTPUT_CHANNELS,)],
    bn_bias: I.In[I.f32, (OUTPUT_CHANNELS,)],
    eps: I.f32,
    output: I.Out[
        I.f32,
        (BATCH, OUTPUT_CHANNELS, OUTPUT_HEIGHT, OUTPUT_WIDTH),
    ],
) -> None:
    output_pixels = OUTPUT_HEIGHT * OUTPUT_WIDTH
    reduction_count = INPUT_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH
    kernel_size = KERNEL_HEIGHT * KERNEL_WIDTH

    output_rows = I.domain(0, output_pixels)
    output_columns = I.domain(0, OUTPUT_CHANNELS)
    reduction_columns = I.domain(0, reduction_count)
    output_channel_indices = I.reshape(
        I.indices(output_columns),
        (1, OUTPUT_CHANNELS),
    )

    row_indices = I.reshape(
        I.indices(output_rows),
        (output_pixels, 1),
    )
    reduction_indices = I.reshape(
        I.indices(reduction_columns),
        (1, reduction_count),
    )

    output_y = row_indices // OUTPUT_WIDTH
    output_x = row_indices % OUTPUT_WIDTH
    input_channels = reduction_indices // kernel_size
    kernel_remainder = reduction_indices % kernel_size
    kernel_y = kernel_remainder // KERNEL_WIDTH
    kernel_x = kernel_remainder % KERNEL_WIDTH
    input_y = output_y * STRIDE - PADDING + kernel_y * DILATION
    input_x = output_x * STRIDE - PADDING + kernel_x * DILATION

    input_patches = input[0, input_channels, input_y, input_x]
    weight_matrix = I.reshape(
        weight,
        (OUTPUT_CHANNELS, reduction_count),
    )
    weight_transposed = I.transpose(weight_matrix, (1, 0))
    convolution = I.matmul(
        input_patches,
        weight_transposed,
        acc_dtype=I.f32,
    )
    convolution = convolution + conv_bias

    variance_epsilon = I.sqrt(running_var + eps)
    normalized = (convolution - running_mean) / variance_epsilon
    activated = I.maximum(
        normalized * bn_weight + bn_bias,
        I.cast(0.0, I.f32),
    )
    output[0, output_channel_indices, output_y, output_x] = activated


def build(context):
    constexprs = {
        "BATCH": 1,
        "INPUT_CHANNELS": 64,
        "INPUT_HEIGHT": 32,
        "INPUT_WIDTH": 32,
        "OUTPUT_CHANNELS": 128,
        "KERNEL_HEIGHT": 3,
        "KERNEL_WIDTH": 3,
        "OUTPUT_HEIGHT": 32,
        "OUTPUT_WIDTH": 32,
        "STRIDE": 1,
        "PADDING": 1,
        "DILATION": 1,
    }
    no_bias_artifact = context.compile(
        "relu_batch_norm_conv2d_no_bias",
        conv_bn_relu_no_bias,
        constexprs=constexprs,
    )
    with_bias_artifact = context.compile(
        "relu_batch_norm_conv2d_with_bias",
        conv_bn_relu_with_bias,
        constexprs=constexprs,
    )

    def relu_batch_norm_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        running_mean=None,
        running_var=None,
        bn_weight=None,
        bn_bias=None,
        training=False,
        momentum=0.1,
        eps=1e-05,
        inplace=False,
    ):
        if (
            stride != 1
            or padding != 1
            or dilation != 1
            or groups != 1
            or training
            or inplace
        ):
            raise ValueError("operation is specialized to the supplied invocation profile")
        if running_mean is None or running_var is None:
            raise ValueError("running_mean and running_var are required")
        if bn_weight is None or bn_bias is None:
            raise ValueError("bn_weight and bn_bias are required")

        if bias is None:
            return no_bias_artifact.run(
                input,
                weight,
                running_mean,
                running_var,
                bn_weight,
                bn_bias,
                eps,
            )
        return with_bias_artifact.run(
            input,
            weight,
            bias,
            running_mean,
            running_var,
            bn_weight,
            bn_bias,
            eps,
        )

    return relu_batch_norm_conv2d
