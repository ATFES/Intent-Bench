from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def _selu_elementwise(
    values: I.In[I.f32, ("N",)],
    result: I.Out[I.f32, ("N",)],
) -> None:
    alpha = I.cast(1.6732632423543772, I.f32)
    scale = I.cast(1.0507009873554805, I.f32)
    zero = I.cast(0.0, I.f32)

    for index in I.parallel(I.domain(0, values.shape[0])):
        value = values[index]
        exponential = I.exp(value)
        softplus = alpha * (exponential - I.cast(1.0, I.f32))
        activated = I.maximum(value, zero) + I.minimum(value, softplus)
        result[index] = scale * activated


def build(context):
    selu_kernel = context.compile("selu_elementwise_v1", _selu_elementwise)

    def selu(input, inplace: bool = False):
        output = input.new_empty(input.shape)
        selu_kernel(input, output)
        return output

    return selu
