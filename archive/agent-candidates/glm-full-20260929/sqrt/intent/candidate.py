from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def sqrt_elementwise(
    values: I.In[I.f32, ("N",)],
    result: I.Out[I.f32, ("N",)],
) -> None:
    for index in I.parallel(I.domain(0, values.shape[0])):
        result[index] = I.sqrt(values[index])


def build(context):
    kernel = context.compile("sqrt_elementwise", sqrt_elementwise)

    def sqrt(input, out=None):
        if out is None:
            return kernel.run(input)
        kernel(input, out)
        return out

    return sqrt
