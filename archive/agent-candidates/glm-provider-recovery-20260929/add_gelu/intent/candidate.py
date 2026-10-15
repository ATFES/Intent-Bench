from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def add_gelu_tensor(
    result: I.InOut[I.f16, ("N",)],
    input: I.In[I.f16, ("N",)],
    other: I.In[I.f16, ("N",)],
    alpha: I.f32,
    APPROXIMATE: I.Constexpr[bool],
) -> None:
    for i in I.parallel(I.domain(0, input.shape[0])):
        x = I.cast(input[i], I.f32) + alpha * I.cast(other[i], I.f32)
        if APPROXIMATE:
            coefficient = I.cast(0.7978845608028654, I.f32)
            inner = coefficient * (x + I.cast(0.044715, I.f32) * x * x * x)
            y = I.cast(0.5, I.f32) * x * (I.cast(1.0, I.f32) + I.tanh(inner, approximate=True))
        else:
            y = I.cast(0.5, I.f32) * x * (I.cast(1.0, I.f32) + I.erf(x * I.cast(0.7071067811865476, I.f32)))
        result[i] = I.cast(y, I.f16)


@intent.kernel
def add_gelu_scalar(
    result: I.InOut[I.f16, ("N",)],
    input: I.In[I.f16, ("N",)],
    other: I.f16,
    alpha: I.f32,
    APPROXIMATE: I.Constexpr[bool],
) -> None:
    for i in I.parallel(I.domain(0, input.shape[0])):
        x = I.cast(input[i], I.f32) + alpha * I.cast(other, I.f32)
        if APPROXIMATE:
            coefficient = I.cast(0.7978845608028654, I.f32)
            inner = coefficient * (x + I.cast(0.044715, I.f32) * x * x * x)
            y = I.cast(0.5, I.f32) * x * (I.cast(1.0, I.f32) + I.tanh(inner, approximate=True))
        else:
            y = I.cast(0.5, I.f32) * x * (I.cast(1.0, I.f32) + I.erf(x * I.cast(0.7071067811865476, I.f32)))
        result[i] = I.cast(y, I.f16)


def build(context):
    tensor_artifacts = {
        approximate: context.compile(
            f"add_gelu_tensor_{approximate}",
            add_gelu_tensor,
            constexprs={"APPROXIMATE": approximate == "tanh"},
        )
        for approximate in ("none", "tanh")
    }
    scalar_artifacts = {
        approximate: context.compile(
            f"add_gelu_scalar_{approximate}",
            add_gelu_scalar,
            constexprs={"APPROXIMATE": approximate == "tanh"},
        )
        for approximate in ("none", "tanh")
    }

    def add_gelu(input, other, alpha=1, approximate="none", out=None):
        if approximate not in tensor_artifacts:
            raise ValueError(f"GELU approximation method not supported: {approximate!r}")
        alpha_value = float(alpha)
        result = torch.empty_like(input) if out is None else out
        if isinstance(other, int):
            scalar_artifacts[approximate](result, input, float(other), alpha_value)
        else:
            tensor_artifacts[approximate](result, input, other, alpha_value)
        return result

    return add_gelu
