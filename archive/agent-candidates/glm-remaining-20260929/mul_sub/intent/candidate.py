from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def mul_sub_tensor_tensor(
    input: I.In[I.f32, ("N",)],
    other_mul: I.In[I.f32, ("N",)],
    other_sub: I.In[I.f32, ("N",)],
    alpha: I.f32,
    out: I.Out[I.f32, ("N",)],
) -> None:
    for element in I.parallel(I.domain(0, input.shape[0])):
        out[element] = (
            input[element] * other_mul[element] - alpha * other_sub[element]
        )


@intent.kernel
def mul_sub_tensor_scalar(
    input: I.In[I.f32, ("N",)],
    other_mul: I.In[I.f32, ("N",)],
    other_sub: I.f32,
    alpha: I.f32,
    out: I.Out[I.f32, ("N",)],
) -> None:
    for element in I.parallel(I.domain(0, input.shape[0])):
        out[element] = input[element] * other_mul[element] - alpha * other_sub


@intent.kernel
def mul_sub_scalar_tensor(
    input: I.In[I.f32, ("N",)],
    other_mul: I.f32,
    other_sub: I.In[I.f32, ("N",)],
    alpha: I.f32,
    out: I.Out[I.f32, ("N",)],
) -> None:
    for element in I.parallel(I.domain(0, input.shape[0])):
        out[element] = input[element] * other_mul - alpha * other_sub[element]


@intent.kernel
def mul_sub_scalar_scalar(
    input: I.In[I.f32, ("N",)],
    other_mul: I.f32,
    other_sub: I.f32,
    alpha: I.f32,
    out: I.Out[I.f32, ("N",)],
) -> None:
    for element in I.parallel(I.domain(0, input.shape[0])):
        out[element] = input[element] * other_mul - alpha * other_sub


def build(context):
    tensor_tensor = context.compile(
        "mul_sub_tensor_tensor", mul_sub_tensor_tensor
    )
    tensor_scalar = context.compile(
        "mul_sub_tensor_scalar", mul_sub_tensor_scalar
    )
    scalar_tensor = context.compile(
        "mul_sub_scalar_tensor", mul_sub_scalar_tensor
    )
    scalar_scalar = context.compile(
        "mul_sub_scalar_scalar", mul_sub_scalar_scalar
    )

    def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
        if out is None:
            output = torch.empty_like(input)
        else:
            output = out

        other_mul_is_tensor = isinstance(other_mul, torch.Tensor)
        other_sub_is_tensor = isinstance(other_sub, torch.Tensor)
        alpha_value = float(alpha)

        if other_mul_is_tensor and other_sub_is_tensor:
            tensor_tensor(input, other_mul, other_sub, alpha_value, output)
        elif other_mul_is_tensor:
            tensor_scalar(input, other_mul, float(other_sub), alpha_value, output)
        elif other_sub_is_tensor:
            scalar_tensor(input, float(other_mul), other_sub, alpha_value, output)
        else:
            scalar_scalar(
                input,
                float(other_mul),
                float(other_sub),
                alpha_value,
                output,
            )

        return output

    return mul_sub
