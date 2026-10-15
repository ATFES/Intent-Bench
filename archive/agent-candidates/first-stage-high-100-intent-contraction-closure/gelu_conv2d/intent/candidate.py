from __future__ import annotations

from typing import Optional, Tuple, Union

import intent
import intent.language as I
import torch


@intent.kernel
def _conv_gelu_no_bias(
    x: I.In[I.f32, (16, 16, 64, 64)],
    weight: I.In[I.f32, (32, 16, 3, 3)],
    out: I.Out[I.f32, (16, 32, 64, 64)],
    TANH: I.Constexpr[bool],
):
    n_domain = I.domain(0, 16)
    co_domain = I.domain(0, 32)
    oh_domain = I.domain(0, 64)
    ow_domain = I.domain(0, 64)
    ci_domain = I.domain(0, 16)

    n_index = I.reshape(I.indices(n_domain), (16, 1, 1, 1, 1))
    co_index = I.reshape(I.indices(co_domain), (1, 32, 1, 1, 1))
    ci_index = I.reshape(I.indices(ci_domain), (1, 1, 16, 1, 1))
    oh_index = I.reshape(I.indices(oh_domain), (1, 1, 1, 64, 1))
    ow_index = I.reshape(I.indices(ow_domain), (1, 1, 1, 1, 64))

    acc = I.zeros((16, 32, 64, 64), I.f32)
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    half = I.cast(0.5, I.f32)

    for kh in I.domain(0, 3):
        for kw in I.domain(0, 3):
            ih = oh_index + kh - 1
            iw = ow_index + kw - 1
            valid = I.select(ih >= 0, ih < 64, False)
            valid = I.select(valid, iw >= 0, False)
            valid = I.select(valid, iw < 64, False)

            x_tile = I.gather(
                x,
                (n_index, ci_index, ih, iw),
                valid=valid,
                fill=zero,
            )
            w_tile = weight[co_index, ci_index, kh, kw]
            partial = I.reduce.sum(x_tile * w_tile, axis=2, acc_dtype=I.f32)
            acc = acc + partial

    if TANH:
        c = I.cast(0.7978845608028654, I.f32)
        cubic = acc * acc * acc
        inner = c * (acc + I.cast(0.044715, I.f32) * cubic)
        gelu = half * acc * (one + I.tanh(inner))
    else:
        sqrt_two = I.sqrt(I.cast(2.0, I.f32))
        gelu = half * acc * (one + I.erf(I.fdiv(acc, sqrt_two)))

    out[n_domain, co_domain, oh_domain, ow_domain] = gelu


@intent.kernel
def _conv_gelu_bias(
    x: I.In[I.f32, (16, 16, 64, 64)],
    weight: I.In[I.f32, (32, 16, 3, 3)],
    bias: I.In[I.f32, (32,)],
    out: I.Out[I.f32, (16, 32, 64, 64)],
    TANH: I.Constexpr[bool],
):
    n_domain = I.domain(0, 16)
    co_domain = I.domain(0, 32)
    oh_domain = I.domain(0, 64)
    ow_domain = I.domain(0, 64)
    ci_domain = I.domain(0, 16)

    n_index = I.reshape(I.indices(n_domain), (16, 1, 1, 1, 1))
    co_index = I.reshape(I.indices(co_domain), (1, 32, 1, 1, 1))
    ci_index = I.reshape(I.indices(ci_domain), (1, 1, 16, 1, 1))
    oh_index = I.reshape(I.indices(oh_domain), (1, 1, 1, 64, 1))
    ow_index = I.reshape(I.indices(ow_domain), (1, 1, 1, 1, 64))

    acc = I.zeros((16, 32, 64, 64), I.f32)
    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    half = I.cast(0.5, I.f32)

    for kh in I.domain(0, 3):
        for kw in I.domain(0, 3):
            ih = oh_index + kh - 1
            iw = ow_index + kw - 1
            valid = I.select(ih >= 0, ih < 64, False)
            valid = I.select(valid, iw >= 0, False)
            valid = I.select(valid, iw < 64, False)

            x_tile = I.gather(
                x,
                (n_index, ci_index, ih, iw),
                valid=valid,
                fill=zero,
            )
            w_tile = weight[co_index, ci_index, kh, kw]
            partial = I.reduce.sum(x_tile * w_tile, axis=2, acc_dtype=I.f32)
            acc = acc + partial

    bias_value = bias[I.reshape(I.indices(co_domain), (1, 32, 1, 1))]
    acc = acc + bias_value

    if TANH:
        c = I.cast(0.7978845608028654, I.f32)
        cubic = acc * acc * acc
        inner = c * (acc + I.cast(0.044715, I.f32) * cubic)
        gelu = half * acc * (one + I.tanh(inner))
    else:
        sqrt_two = I.sqrt(I.cast(2.0, I.f32))
        gelu = half * acc * (one + I.erf(I.fdiv(acc, sqrt_two)))

    out[n_domain, co_domain, oh_domain, ow_domain] = gelu


def build(context):
    no_bias_none = context.compile(
        "gelu_conv2d_no_bias_none",
        _conv_gelu_no_bias,
        constexprs={"TANH": False},
    )
    no_bias_tanh = context.compile(
        "gelu_conv2d_no_bias_tanh",
        _conv_gelu_no_bias,
        constexprs={"TANH": True},
    )
    bias_none = context.compile(
        "gelu_conv2d_bias_none",
        _conv_gelu_bias,
        constexprs={"TANH": False},
    )
    bias_tanh = context.compile(
        "gelu_conv2d_bias_tanh",
        _conv_gelu_bias,
        constexprs={"TANH": True},
    )

    def gelu_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int], str] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        approximate: str = "none",
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        def pair(value):
            if isinstance(value, int):
                return (value, value)
            return tuple(value)

        if pair(stride) != (1, 1) or pair(dilation) != (1, 1):
            raise NotImplementedError("the supplied profile uses unit stride and dilation")
        if padding != 1 and padding != "same" and pair(padding) != (1, 1):
            raise NotImplementedError("the supplied profile uses one-cell padding")
        if groups != 1:
            raise NotImplementedError("the supplied profile uses one convolution group")
        if approximate not in ("none", "tanh"):
            raise ValueError("approximate must be 'none' or 'tanh'")

        if bias is None:
            artifact = no_bias_tanh if approximate == "tanh" else no_bias_none
            if out is None:
                return artifact.run(input, weight)
            artifact(input, weight, out)
            return out

        artifact = bias_tanh if approximate == "tanh" else bias_none
        if out is None:
            return artifact.run(input, weight, bias)
        artifact(input, weight, bias, out)
        return out

    return gelu_conv2d
