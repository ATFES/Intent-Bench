from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def _softplus_linear_without_bias(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    output: I.Out[I.f32, ("M", "N")],
    beta: I.f32,
    threshold: I.f32,
) -> None:
    z = I.matmul(input, weight, transpose_rhs=True, acc_dtype=I.f32)
    softplus = I.log1p(I.exp(beta * z)) / beta
    output[:, :] = I.select(z > threshold, z, softplus)


@intent.kernel
def _softplus_linear_with_bias(
    input: I.In[I.f32, ("M", "K")],
    weight: I.In[I.f32, ("N", "K")],
    bias: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("M", "N")],
    beta: I.f32,
    threshold: I.f32,
) -> None:
    z = I.matmul(input, weight, transpose_rhs=True, acc_dtype=I.f32) + bias
    softplus = I.log1p(I.exp(beta * z)) / beta
    output[:, :] = I.select(z > threshold, z, softplus)


def build(context):
    without_bias = context.compile(
        "softplus_linear_without_bias", _softplus_linear_without_bias
    )
    with_bias = context.compile(
        "softplus_linear_with_bias", _softplus_linear_with_bias
    )

    def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
        output = torch.empty(
            (input.shape[0], weight.shape[0]),
            dtype=input.dtype,
            device=input.device,
        )
        beta_value = float(beta)
        threshold_value = float(threshold)
        if bias is None:
            without_bias(
                input,
                weight,
                output,
                beta_value,
                threshold_value,
            )
        else:
            with_bias(
                input,
                weight,
                bias,
                output,
                beta_value,
                threshold_value,
            )
        return output

    return softplus_linear
