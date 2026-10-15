from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import intent
import intent.language as I
import torch


Pair = Union[int, Tuple[int, int]]


def _pair(value: Pair, name: str) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(component, int) for component in value)
    ):
        return value
    raise ValueError(f"{name} must be an int or a pair of ints")


def _padding(value: Union[Pair, str]) -> Tuple[int, int]:
    if value == "same":
        return -1, -1
    if value == "valid":
        return 0, 0
    return _pair(value, "padding")


@intent.kernel
def _im2col(
    input: I.In[I.f32, ("B", "C", "IH", "IW")],
    columns: I.Out[I.f32, ("M", "K")],
    stride_h: I.i32,
    stride_w: I.i32,
    padding_h: I.i32,
    padding_w: I.i32,
    dilation_h: I.i32,
    dilation_w: I.i32,
    kernel_h: I.i32,
    kernel_w: I.i32,
) -> None:
    stride_hi = I.cast(stride_h, I.index)
    stride_wi = I.cast(stride_w, I.index)
    padding_hi = I.cast(padding_h, I.index)
    padding_wi = I.cast(padding_w, I.index)
    dilation_hi = I.cast(dilation_h, I.index)
    dilation_wi = I.cast(dilation_w, I.index)
    kernel_hi = I.cast(kernel_h, I.index)
    kernel_wi = I.cast(kernel_w, I.index)
    output_height = (
        input.shape[2] + 2 * padding_hi - dilation_hi * (kernel_hi - 1) - 1
    ) // stride_hi + 1
    output_width = (
        input.shape[3] + 2 * padding_wi - dilation_wi * (kernel_wi - 1) - 1
    ) // stride_wi + 1

    for row in I.parallel(I.domain(0, columns.shape[0])):
        position = row % output_width
        spatial = row // output_width
        output_y = spatial % output_height
        batch = spatial // output_height

        for feature in I.parallel(I.domain(0, columns.shape[1])):
            kernel_x = feature % kernel_wi
            channel_kernel = feature // kernel_wi
            kernel_y = channel_kernel % kernel_hi
            channel = channel_kernel // kernel_hi

            input_y = output_y * stride_hi - padding_hi + kernel_y * dilation_hi
            input_x = position * stride_wi - padding_wi + kernel_x * dilation_wi
            valid = (
                (batch < input.shape[0])
                & (channel < input.shape[1])
                & (input_y >= 0)
                & (input_y < input.shape[2])
                & (input_x >= 0)
                & (input_x < input.shape[3])
            )
            columns[row, feature] = I.gather(
                input,
                (batch, channel, input_y, input_x),
                valid=valid,
                fill=0.0,
            )


@intent.kernel
def _conv_gelu(
    columns: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("O", "K")],
    bias: I.In[I.f32, ("O",)],
    output: I.Out[I.f32, ("M", "O")],
    HAS_BIAS: I.Constexpr[bool],
    USE_TANH: I.Constexpr[bool],
) -> None:
    result = I.matmul(columns[:, :], I.transpose(weight[:, :]), acc_dtype=I.f32)
    if HAS_BIAS:
        result = result + bias[:]

    if USE_TANH:
        argument = I.cast(0.044715, I.f32) * result * result * result + result
        argument = argument * I.cast(0.7978845608028654, I.f32)
        result = I.cast(0.5, I.f32) * result * (I.cast(1.0, I.f32) + I.tanh(argument))
    else:
        probability = I.erf(result * I.cast(0.7071067811865476, I.f32))
        result = I.cast(0.5, I.f32) * result * (I.cast(1.0, I.f32) + probability)

    output[:, :] = result


def _output_size(size: int, padding: int, dilation: int, kernel_size: int, stride: int) -> int:
    return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def build(context) -> Callable[..., torch.Tensor]:
    im2col = context.compile(
        "gelu_conv2d_im2col",
        _im2col,
    )
    gelu_no_bias = context.compile(
        "gelu_conv2d_gemm_no_bias_erf",
        _conv_gelu,
        constexprs={"HAS_BIAS": False, "USE_TANH": False},
    )
    tanh_no_bias = context.compile(
        "gelu_conv2d_gemm_no_bias_tanh",
        _conv_gelu,
        constexprs={"HAS_BIAS": False, "USE_TANH": True},
    )
    gelu_bias = context.compile(
        "gelu_conv2d_gemm_bias_erf",
        _conv_gelu,
        constexprs={"HAS_BIAS": True, "USE_TANH": False},
    )
    tanh_bias = context.compile(
        "gelu_conv2d_gemm_bias_tanh",
        _conv_gelu,
        constexprs={"HAS_BIAS": True, "USE_TANH": True},
    )
    variants = {
        (False, False): gelu_no_bias,
        (False, True): tanh_no_bias,
        (True, False): gelu_bias,
        (True, True): tanh_bias,
    }

    def gelu_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Pair = 1,
        padding: Union[Pair, str] = 0,
        dilation: Pair = 1,
        groups: int = 1,
        approximate: str = "none",
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input.ndim != 4 or weight.ndim != 4:
            raise ValueError("input and weight must both be rank four")
        if groups != 1:
            raise NotImplementedError("only groups=1 is supported by this profile")
        if input.dtype != torch.float32 or weight.dtype != torch.float32:
            raise TypeError("input and weight must be float32")
        if approximate not in ("none", "tanh"):
            raise ValueError("approximate must be 'none' or 'tanh'")

        stride_h, stride_w = _pair(stride, "stride")
        dilation_h, dilation_w = _pair(dilation, "dilation")
        padding_h, padding_w = _padding(padding)
        if padding_h < 0 or padding_w < 0:
            input_h, input_w = input.shape[2], input.shape[3]
            output_h = (input_h + stride_h - 1) // stride_h
            output_w = (input_w + stride_w - 1) // stride_w
            padding_h = max(
                0,
                (output_h - 1) * stride_h
                + dilation_h * (weight.shape[2] - 1)
                + 1
                - input_h,
            )
            padding_w = max(
                0,
                (output_w - 1) * stride_w
                + dilation_w * (weight.shape[3] - 1)
                + 1
                - input_w,
            )

        batch, in_channels, input_h, input_w = input.shape
        out_channels, weight_channels, weight_h, weight_w = weight.shape
        if in_channels != weight_channels:
            raise ValueError("input and weight channel extents do not match")
        if bias is not None and bias.shape != (out_channels,):
            raise ValueError("bias has the wrong shape")

        output_h = _output_size(input_h, padding_h, dilation_h, weight_h, stride_h)
        output_w = _output_size(input_w, padding_w, dilation_w, weight_w, stride_w)
        output = (
            out
            if out is not None
            else torch.empty((batch, out_channels, output_h, output_w), device=input.device, dtype=input.dtype)
        )
        if output.shape != (batch, out_channels, output_h, output_w) or output.dtype != input.dtype:
            raise ValueError("out has the wrong shape or dtype")

        row_count = batch * output_h * output_w
        feature_count = in_channels * weight_h * weight_w
        columns = torch.empty((row_count, feature_count), device=input.device, dtype=input.dtype)
        output_rows = output.view(row_count, out_channels)
        kernel_bias = bias if bias is not None else torch.empty((out_channels,), device=input.device, dtype=input.dtype)

        im2col(
            input,
            columns,
            stride_h,
            stride_w,
            padding_h,
            padding_w,
            dilation_h,
            dilation_w,
            weight_h,
            weight_w,
        )
        variants[(bias is not None, approximate == "tanh")](
            columns,
            weight,
            kernel_bias,
            output_rows,
        )
        return output

    return gelu_conv2d
