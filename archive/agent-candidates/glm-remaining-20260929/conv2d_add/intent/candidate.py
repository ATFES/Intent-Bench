from __future__ import annotations

from typing import Callable

import intent
import intent.language as I


def _pair(value: object, name: str) -> tuple[int, int]:
    if value == "valid":
        return (0, 0)
    if value == "same":
        raise ValueError("padding='same' requires the spatial sizes")
    if isinstance(value, int):
        return (value, value)
    try:
        parsed = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an int, pair, 'valid', or 'same'") from exc
    if len(parsed) != 2:
        raise ValueError(f"{name} must contain two values")
    return (int(parsed[0]), int(parsed[1]))


def _convolution_size(
    size: int, padding: int, dilation: int, kernel_size: int, stride: int
) -> int:
    if min(size, stride, padding, dilation, kernel_size) < 0:
        raise ValueError("convolution sizes and parameters must be nonnegative")
    return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def _same_padding(size: int, dilation: int, kernel_size: int) -> int:
    return (dilation * (kernel_size - 1)) // 2


@intent.kernel
def _conv2d_add(
    input: I.In[I.f32, ("B", "C", "IH", "IW")],
    weight: I.In[I.f32, ("O", "C", "KH", "KW")],
    bias: I.In[I.f32, ("O",)],
    other: I.In[I.f32, ("B", "O", "OH", "OW")],
    alpha: I.f64,
    other_scalar: I.f64,
    out: I.Out[I.f32, ("B", "O", "OH", "OW")],
    STRIDE_H: I.Constexpr[int],
    STRIDE_W: I.Constexpr[int],
    PAD_H: I.Constexpr[int],
    PAD_W: I.Constexpr[int],
    DILATION_H: I.Constexpr[int],
    DILATION_W: I.Constexpr[int],
    GROUPS: I.Constexpr[int],
    HAS_BIAS: I.Constexpr[bool],
    OTHER_MODE: I.Constexpr[int],
):
    batch_count = input.shape[0]
    input_channels_per_group = weight.shape[1]
    output_channels = out.shape[1]
    output_height = out.shape[2]
    output_width = out.shape[3]
    output_channels_per_group = output_channels // GROUPS
    kernel_height = weight.shape[2]
    kernel_width = weight.shape[3]

    batches = I.domain(0, batch_count)
    output_channels_domain = I.domain(0, output_channels)
    output_rows = I.domain(0, output_height)
    output_columns = I.domain(0, output_width)
    input_channels = I.domain(0, input_channels_per_group)
    kernel_rows = I.domain(0, kernel_height)
    kernel_columns = I.domain(0, kernel_width)

    batch_index = I.indices(batches)
    output_channel_index = I.indices(output_channels_domain)
    output_row_index = I.indices(output_rows)
    output_column_index = I.indices(output_columns)
    input_channel_index = I.indices(input_channels)
    kernel_row_index = I.indices(kernel_rows)
    kernel_column_index = I.indices(kernel_columns)

    batch_coordinates = I.reshape(
        batch_index,
        (batch_index.shape[0], 1, 1, 1, 1, 1, 1),
    )
    output_channel_coordinates = I.reshape(
        output_channel_index,
        (1, 1, 1, output_channel_index.shape[0], 1, 1, 1),
    )
    output_row_coordinates = I.reshape(
        output_row_index,
        (1, 1, output_row_index.shape[0], 1, 1, 1, 1),
    )
    output_column_coordinates = I.reshape(
        output_column_index,
        (1, 1, 1, output_column_index.shape[0], 1, 1, 1),
    )
    local_input_channel_coordinates = I.reshape(
        input_channel_index,
        (1, 1, 1, 1, input_channel_index.shape[0], 1, 1),
    )
    group_index = output_channel_coordinates // I.cast(output_channels_per_group, I.index)
    input_channel_coordinates = (
        group_index * I.cast(input_channels_per_group, I.index)
        + local_input_channel_coordinates
    )
    kernel_row_coordinates = I.reshape(
        kernel_row_index,
        (1, 1, 1, 1, 1, kernel_row_index.shape[0], 1),
    )
    kernel_column_coordinates = I.reshape(
        kernel_column_index,
        (1, 1, 1, 1, 1, 1, kernel_column_index.shape[0]),
    )

    input_row_coordinates = (
        output_row_coordinates * I.cast(STRIDE_H, I.index)
        - I.cast(PAD_H, I.index)
        + kernel_row_coordinates * I.cast(DILATION_H, I.index)
    )
    input_column_coordinates = (
        output_column_coordinates * I.cast(STRIDE_W, I.index)
        - I.cast(PAD_W, I.index)
        + kernel_column_coordinates * I.cast(DILATION_W, I.index)
    )
    input_window = input[
        batch_coordinates,
        input_channel_coordinates,
        input_row_coordinates,
        input_column_coordinates,
    ]

    weight_output_coordinates = output_channel_coordinates % I.cast(
        output_channels_per_group, I.index
    )
    filters = weight[
        weight_output_coordinates,
        local_input_channel_coordinates,
        kernel_row_coordinates,
        kernel_column_coordinates,
    ]
    convolution = I.contract(
        input_window,
        filters,
        reduce=((4, 4), (5, 5), (6, 6)),
        acc_dtype=I.f32,
    )
    result = convolution
    if HAS_BIAS:
        result = result + bias
    if OTHER_MODE == 1:
        batch_output_coordinates = I.reshape(
            batch_index,
            (batch_index.shape[0], 1, 1, 1),
        )
        output_channel_output_coordinates = I.reshape(
            output_channel_index,
            (1, 1, 1, output_channel_index.shape[0]),
        )
        output_row_output_coordinates = I.reshape(
            output_row_index,
            (1, output_row_index.shape[0], 1, 1),
        )
        output_column_output_coordinates = I.reshape(
            output_column_index,
            (1, 1, output_column_index.shape[0], 1),
        )
        other_value = other[
            batch_output_coordinates,
            output_channel_output_coordinates,
            output_row_output_coordinates,
            output_column_output_coordinates,
        ]
        result = result + I.cast(alpha, I.f32) * other_value
    if OTHER_MODE == 2:
        result = result + I.cast(alpha, I.f32) * I.cast(other_scalar, I.f32)

    batch_store_coordinates = I.reshape(
        batch_index,
        (batch_index.shape[0], 1, 1, 1),
    )
    output_channel_store_coordinates = I.reshape(
        output_channel_index,
        (1, 1, 1, output_channel_index.shape[0]),
    )
    output_row_store_coordinates = I.reshape(
        output_row_index,
        (1, output_row_index.shape[0], 1, 1),
    )
    output_column_store_coordinates = I.reshape(
        output_column_index,
        (1, 1, output_column_index.shape[0], 1),
    )
    out[
        batch_store_coordinates,
        output_channel_store_coordinates,
        output_row_store_coordinates,
        output_column_store_coordinates,
    ] = result


def build(context) -> Callable:
    stride_h = 1
    stride_w = 1
    pad_h = 1
    pad_w = 1
    dilation_h = 1
    dilation_w = 1
    groups = 1
    constants = {
        "STRIDE_H": stride_h,
        "STRIDE_W": stride_w,
        "PAD_H": pad_h,
        "PAD_W": pad_w,
        "DILATION_H": dilation_h,
        "DILATION_W": dilation_w,
        "GROUPS": groups,
    }

    artifacts = {
        (False, 0): context.compile(
            "conv2d_add_no_bias_no_other",
            _conv2d_add,
            constexprs={**constants, "HAS_BIAS": False, "OTHER_MODE": 0},
        ),
        (False, 1): context.compile(
            "conv2d_add_no_bias_tensor_other",
            _conv2d_add,
            constexprs={**constants, "HAS_BIAS": False, "OTHER_MODE": 1},
        ),
        (False, 2): context.compile(
            "conv2d_add_no_bias_scalar_other",
            _conv2d_add,
            constexprs={**constants, "HAS_BIAS": False, "OTHER_MODE": 2},
        ),
        (True, 0): context.compile(
            "conv2d_add_bias_no_other",
            _conv2d_add,
            constexprs={**constants, "HAS_BIAS": True, "OTHER_MODE": 0},
        ),
        (True, 1): context.compile(
            "conv2d_add_bias_tensor_other",
            _conv2d_add,
            constexprs={**constants, "HAS_BIAS": True, "OTHER_MODE": 1},
        ),
        (True, 2): context.compile(
            "conv2d_add_bias_scalar_other",
            _conv2d_add,
            constexprs={**constants, "HAS_BIAS": True, "OTHER_MODE": 2},
        ),
    }

    def conv2d_add(
        input,
        weight,
        bias=None,
        other=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        alpha=1,
        out=None,
    ):
        if input.ndim != 4 or weight.ndim != 4:
            raise ValueError("input and weight must both be four dimensional")
        if groups <= 0:
            raise ValueError("groups must be positive")
        if input.shape[1] % groups != 0 or weight.shape[0] % groups != 0:
            raise ValueError("groups must divide input and output channels")
        if weight.shape[1] != input.shape[1] // groups:
            raise ValueError("weight input channels do not match input/groups")

        stride_h, stride_w = _pair(stride, "stride")
        dilation_h, dilation_w = _pair(dilation, "dilation")
        if padding == "same":
            if stride_h != 1 or stride_w != 1:
                raise ValueError("padding='same' requires unit stride")
            pad_h = _same_padding(input.shape[2], dilation_h, weight.shape[2])
            pad_w = _same_padding(input.shape[3], dilation_w, weight.shape[3])
        else:
            pad_h, pad_w = _pair(padding, "padding")

        output_height = _convolution_size(
            input.shape[2], pad_h, dilation_h, weight.shape[2], stride_h
        )
        output_width = _convolution_size(
            input.shape[3], pad_w, dilation_w, weight.shape[3], stride_w
        )
        output_shape = (input.shape[0], weight.shape[0], output_height, output_width)
        if out is None:
            output = input.new_empty(output_shape)
        else:
            if tuple(out.shape) != output_shape:
                raise ValueError("out has an incompatible shape")
            output = out

        profile_matches = (
            stride_h == 1
            and stride_w == 1
            and pad_h == 1
            and pad_w == 1
            and dilation_h == 1
            and dilation_w == 1
            and groups == 1
        )
        if not profile_matches:
            raise ValueError("this compiled candidate supports the supplied profile only")

        if bias is None:
            has_bias = False
            bias_view = weight[:, 0, 0, 0]
        else:
            if tuple(bias.shape) != (weight.shape[0],):
                raise ValueError("bias must have shape (out_channels,)")
            has_bias = True
            bias_view = bias

        if other is None:
            other_mode = 0
            other_view = output
        elif isinstance(other, (int, float)):
            other_mode = 2
            other_view = output
        else:
            if tuple(other.shape) != output_shape:
                raise ValueError("other must have the output shape")
            other_mode = 1
            other_view = other

        artifacts[(has_bias, other_mode)](
            input,
            weight,
            bias_view,
            other_view,
            float(alpha),
            float(other) if other is not None else 0.0,
            output,
        )
        return output

    return conv2d_add
