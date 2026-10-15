from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def _gelu_exact(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",), I.constraints(noalias=True)],
) -> None:
    for index in I.parallel(I.domain(0, input.shape[0])):
        value = input[index]
        normal_cdf = I.cast(0.5, I.f32) * (
            I.cast(1.0, I.f32)
            + I.erf(value / I.cast(1.4142135623730951, I.f32))
        )
        output[index] = value * normal_cdf


@intent.kernel
def _gelu_tanh(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",), I.constraints(noalias=True)],
) -> None:
    for index in I.parallel(I.domain(0, input.shape[0])):
        value = input[index]
        argument = I.cast(0.7978845608028654, I.f32) * (
            value + I.cast(0.044715, I.f32) * value * value * value
        )
        output[index] = I.cast(0.5, I.f32) * value * (
            I.cast(1.0, I.f32) + I.tanh(argument)
        )


def build(context):
    exact = context.compile("gelu_exact", _gelu_exact)
    tanh = context.compile("gelu_tanh", _gelu_tanh)

    def gelu(input, approximate: str = "none"):
        output = input.new_empty(input.shape, dtype=input.dtype)
        if approximate == "none":
            exact(input, output)
        elif approximate == "tanh":
            tanh(input, output)
        else:
            raise ValueError("approximate must be 'none' or 'tanh'")
        return output

    return gelu
