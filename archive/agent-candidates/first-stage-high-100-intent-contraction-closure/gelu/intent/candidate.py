import torch
import intent
import intent.language as I


@intent.kernel
def _gelu_none(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    points = I.domain(0, n)
    value = input[points]
    half = I.cast(0.5, I.f32)
    inv_sqrt_2 = I.cast(0.7071067811865475, I.f32)
    output[points] = value * half * (I.cast(1.0, I.f32) + I.erf(value * inv_sqrt_2))


@intent.kernel
def _gelu_tanh(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    points = I.domain(0, n)
    value = input[points]
    half = I.cast(0.5, I.f32)
    one = I.cast(1.0, I.f32)
    coefficient = I.cast(0.044715, I.f32)
    scale = I.cast(0.7978845608028654, I.f32)
    cubic = value * value * value
    output[points] = half * value * (one + I.tanh(scale * (value + coefficient * cubic)))


def build(context):
    none_artifact = context.compile("gelu_none", _gelu_none, constexprs={})
    tanh_artifact = context.compile("gelu_tanh", _gelu_tanh, constexprs={})

    def gelu(input: torch.Tensor, approximate: str = "none") -> torch.Tensor:
        output = torch.empty_like(input)
        if approximate == "none":
            none_artifact(input, output)
        elif approximate == "tanh":
            tanh_artifact(input, output)
        else:
            raise ValueError("approximate must be 'none' or 'tanh'")
        return output

    return gelu
