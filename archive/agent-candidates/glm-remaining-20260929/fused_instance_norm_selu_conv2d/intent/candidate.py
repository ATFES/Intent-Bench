from __future__ import annotations

import intent
import intent.language as I


SELU_ALPHA = 1.6732632423543772848170429916717
SELU_SCALE = 1.0507009873554804934193349852946


@intent.kernel
def im2col(
    source: I.In[I.f32, ("BATCH", "CHANNELS", "HEIGHT", "WIDTH")],
    columns: I.Out[I.f32, ("BATCH", "TAPS", "POSITIONS"), I.constraints(noalias=True)],
    stride_h: I.i64,
    stride_w: I.i64,
    padding_h: I.i64,
    padding_w: I.i64,
    dilation_h: I.i64,
    dilation_w: I.i64,
    output_width: I.i64,
    KERNEL_HEIGHT: I.Constexpr[int],
    KERNEL_WIDTH: I.Constexpr[int],
):
    tap_count = columns.shape[1]
    position_count = columns.shape[2]
    taps = I.indices(I.domain(0, tap_count))
    positions = I.indices(I.domain(0, position_count))

    stride_h_index = I.cast(stride_h, I.index)
    stride_w_index = I.cast(stride_w, I.index)
    padding_h_index = I.cast(padding_h, I.index)
    padding_w_index = I.cast(padding_w, I.index)
    dilation_h_index = I.cast(dilation_h, I.index)
    dilation_w_index = I.cast(dilation_w, I.index)
    output_width_index = I.cast(output_width, I.index)

    kernel_size = KERNEL_HEIGHT * KERNEL_WIDTH
    channel = I.reshape(taps // kernel_size, (tap_count, 1))
    tap_remainder = taps % kernel_size
    kernel_row = I.reshape(tap_remainder // KERNEL_WIDTH, (tap_count, 1))
    kernel_column = I.reshape(tap_remainder % KERNEL_WIDTH, (tap_count, 1))

    row_position = I.reshape(positions // output_width_index, (1, position_count))
    column_position = I.reshape(positions % output_width_index, (1, position_count))
    source_row = (
        row_position * stride_h_index - padding_h_index + kernel_row * dilation_h_index
    )
    source_column = (
        column_position * stride_w_index - padding_w_index + kernel_column * dilation_w_index
    )
    valid = (source_row >= 0) & (source_row < source.shape[2]) & (
        source_column >= 0
    ) & (source_column < source.shape[3]) & (
        channel < source.shape[1]
    )

    for batch in I.parallel(I.domain(0, source.shape[0])):
        columns[batch] = I.gather(
            source,
            (batch, channel, source_row, source_column),
            valid=valid,
            fill=I.cast(0.0, I.f32),
        )


@intent.kernel
def fused_with_bias(
    columns: I.In[I.f32, ("BATCH", "TAPS", "POSITIONS")],
    weight: I.In[I.f32, ("OUT_CHANNELS", "CHANNELS", "WEIGHT_HEIGHT", "WEIGHT_WIDTH")],
    bias: I.In[I.f32, ("OUT_CHANNELS",)],
    normalized: I.Out[I.f32, ("BATCH", "OUT_CHANNELS", "POSITIONS"), I.constraints(noalias=True)],
    epsilon: I.f32,
    SELU_ALPHA: I.Constexpr[float],
    SELU_SCALE: I.Constexpr[float],
):
    flat_weight = I.reshape(
        weight,
        (
            weight.shape[0],
            weight.shape[1] * weight.shape[2] * weight.shape[3],
        ),
    )
    channel_bias = I.reshape(bias, (bias.shape[0], 1))
    alpha = I.cast(SELU_ALPHA, I.f32)
    scale = I.cast(SELU_SCALE, I.f32)
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)

    for batch in I.parallel(I.domain(0, columns.shape[0])):
        activated = I.matmul(
            flat_weight,
            columns[batch],
            acc_dtype=I.f32,
        ) + channel_bias
        activated = I.select(
            activated > zero,
            scale * activated,
            scale * (alpha * (I.exp(activated) - one)),
        )

        position_count = activated.shape[1]
        count = I.cast(position_count, I.f32)
        for channel in I.parallel(I.domain(0, activated.shape[0])):
            values = activated[channel]
            mean = I.reduce.sum(values, axis=0, acc_dtype=I.f32) / count
            centered = values - mean
            variance = (
                I.reduce.sum(centered * centered, axis=0, acc_dtype=I.f32)
                / count
            )
            normalized[batch, channel] = centered * I.rsqrt(variance + epsilon)


@intent.kernel
def fused_without_bias(
    columns: I.In[I.f32, ("BATCH", "TAPS", "POSITIONS")],
    weight: I.In[I.f32, ("OUT_CHANNELS", "CHANNELS", "WEIGHT_HEIGHT", "WEIGHT_WIDTH")],
    normalized: I.Out[I.f32, ("BATCH", "OUT_CHANNELS", "POSITIONS"), I.constraints(noalias=True)],
    epsilon: I.f32,
    SELU_ALPHA: I.Constexpr[float],
    SELU_SCALE: I.Constexpr[float],
):
    flat_weight = I.reshape(
        weight,
        (
            weight.shape[0],
            weight.shape[1] * weight.shape[2] * weight.shape[3],
        ),
    )
    alpha = I.cast(SELU_ALPHA, I.f32)
    scale = I.cast(SELU_SCALE, I.f32)
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)

    for batch in I.parallel(I.domain(0, columns.shape[0])):
        activated = I.matmul(flat_weight, columns[batch], acc_dtype=I.f32)
        activated = I.select(
            activated > zero,
            scale * activated,
            scale * (alpha * (I.exp(activated) - one)),
        )

        position_count = activated.shape[1]
        count = I.cast(position_count, I.f32)
        for channel in I.parallel(I.domain(0, activated.shape[0])):
            values = activated[channel]
            mean = I.reduce.sum(values, axis=0, acc_dtype=I.f32) / count
            centered = values - mean
            variance = (
                I.reduce.sum(centered * centered, axis=0, acc_dtype=I.f32)
                / count
            )
            normalized[batch, channel] = centered * I.rsqrt(variance + epsilon)


def _pair(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("stride, padding, and dilation must have two values")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def build(context):
    im2col_kernel = context.compile(
        "instance_selu_conv_im2col",
        im2col,
        constexprs={
            "KERNEL_HEIGHT": 3,
            "KERNEL_WIDTH": 3,
        },
    )
    with_bias_kernel = context.compile(
        "instance_selu_conv_with_bias",
        fused_with_bias,
        constexprs={
            "SELU_ALPHA": SELU_ALPHA,
            "SELU_SCALE": SELU_SCALE,
        },
    )
    without_bias_kernel = context.compile(
        "instance_selu_conv_without_bias",
        fused_without_bias,
        constexprs={
            "SELU_ALPHA": SELU_ALPHA,
            "SELU_SCALE": SELU_SCALE,
        },
    )

    def fused_instance_norm_selu_conv2d(
        input,
        weight,
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
    ):
        if input.ndim != 4 or weight.ndim != 4:
            raise ValueError("input and weight must be four-dimensional")
        if groups != 1:
            raise ValueError("this invocation requires groups=1")
        if affine or track_running_stats:
            raise ValueError("this invocation requires affine=False and track_running_stats=False")
        if input.shape[1] != weight.shape[1]:
            raise ValueError("input and weight channels do not match")
        if bias is not None and bias.shape[0] != weight.shape[0]:
            raise ValueError("bias and weight output channels do not match")

        batch_size = input.shape[0]
        input_height = input.shape[2]
        input_width = input.shape[3]
        stride_h, stride_w = _pair(stride)
        padding_h, padding_w = _pair(padding)
        dilation_h, dilation_w = _pair(dilation)
        kernel_height = weight.shape[2]
        kernel_width = weight.shape[3]
        output_height = (
            input_height + 2 * padding_h - dilation_h * (kernel_height - 1) - 1
        ) // stride_h + 1
        output_width = (
            input_width + 2 * padding_w - dilation_w * (kernel_width - 1) - 1
        ) // stride_w + 1
        if output_height <= 0 or output_width <= 0:
            raise ValueError("convolution output is empty")

        position_count = output_height * output_width
        tap_count = weight.shape[1] * kernel_height * kernel_width
        columns = im2col_kernel.run(
            input,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            output_width,
        )
        epsilon = float(eps)
        if bias is None:
            output = without_bias_kernel.run(columns, weight, epsilon)
        else:
            output = with_bias_kernel.run(columns, weight, bias, epsilon)
        return output.reshape(
            batch_size,
            weight.shape[0],
            output_height,
            output_width,
        )

    return fused_instance_norm_selu_conv2d
