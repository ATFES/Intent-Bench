import torch
import intent
import intent.language as I


@intent.kernel
def _add_gelu_none(
    input: I.In[I.f16, ("N",)],
    output: I.Out[I.f16, ("N",)],
    other: I.In[I.f16, ("N",)],
    alpha: I.f32,
):
    values = I.domain(0, input.shape[0])
    alpha_f16 = I.cast(alpha, I.f16)
    combined = input[values] + alpha_f16 * other[values]
    combined_f32 = I.cast(combined, I.f32)
    scaled = combined_f32 * I.cast(0.7071067811865475, I.f32)
    gelu = combined_f32 * (
        I.cast(0.5, I.f32) * (I.cast(1.0, I.f32) + I.erf(scaled))
    )
    output[values] = I.cast(gelu, I.f16)


@intent.kernel
def _add_gelu_tanh(
    input: I.In[I.f16, ("N",)],
    output: I.Out[I.f16, ("N",)],
    other: I.In[I.f16, ("N",)],
    alpha: I.f32,
):
    values = I.domain(0, input.shape[0])
    alpha_f16 = I.cast(alpha, I.f16)
    combined = input[values] + alpha_f16 * other[values]
    combined_f32 = I.cast(combined, I.f32)
    cubic = combined_f32 * combined_f32 * combined_f32
    inner = combined_f32 + I.cast(0.044715, I.f32) * cubic
    tanh_arg = I.cast(0.7978845608028654, I.f32) * inner
    gelu = I.cast(0.5, I.f32) * combined_f32 * (
        I.cast(1.0, I.f32) + I.tanh(tanh_arg)
    )
    output[values] = I.cast(gelu, I.f16)


def build(context):
    none_artifact = context.compile("add_gelu_none", _add_gelu_none)
    tanh_artifact = context.compile("add_gelu_tanh", _add_gelu_tanh)

    def add_gelu(input, other, alpha=1, approximate="none", out=None):
        if approximate == "none":
            artifact = none_artifact
        elif approximate == "tanh":
            artifact = tanh_artifact
        else:
            raise ValueError("approximate must be 'none' or 'tanh'")

        alpha_value = float(alpha)
        if out is None:
            return artifact.run(input, other, alpha_value)
        artifact(input, out, other, alpha_value)
        return out

    return add_gelu
