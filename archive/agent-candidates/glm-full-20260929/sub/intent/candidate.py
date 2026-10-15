from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def subtract_contiguous_f32(
    source: I.In[I.f32, ("N",)],
    operand: I.In[I.f32, ("N",)],
    destination: I.Out[I.f32, ("N",)],
    alpha: I.f32,
) -> None:
    for element in I.parallel(I.domain(0, source.shape[0])):
        destination[element] = source[element] - alpha * operand[element]


def build(context):
    subtract = context.compile("subtract_contiguous_f32", subtract_contiguous_f32)

    def sub(input, other, *, alpha=1, out=None):
        if out is None:
            return subtract.run(input, other, alpha)
        subtract(input, other, out, alpha)
        return out

    return sub
