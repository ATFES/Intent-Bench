from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def exp_sqrt_kernel(input: I.In[I.f32, ("N",)], output: I.Out[I.f32, ("N",)]) -> None:
    for index in I.parallel(I.domain(0, input.shape[0])):
        value = I.exp(input[index])
        output[index] = I.sqrt(value)


def build(context):
    exp_sqrt = context.compile("exp_sqrt_elementwise", exp_sqrt_kernel)

    def exp_sqrt(input, out=None):
        if out is None:
            return exp_sqrt.run(input)
        exp_sqrt(input, out)
        return out

    return exp_sqrt
