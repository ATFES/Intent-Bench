from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def bitwise_and_i32(
    input: I.In[I.i32, ("N",)],
    other: I.In[I.i32, ("N",)],
    out: I.Out[I.i32, ("N",)],
) -> None:
    for index in I.parallel(I.domain(0, input.shape[0])):
        out[index] = input[index] & other[index]


def build(context):
    bitwise_and = context.compile("bitwise_and_i32", bitwise_and_i32)

    def bitwise_and_wrapper(input, other, out=None):
        if out is None:
            return bitwise_and.run(input, other)
        return bitwise_and(input, other, out)

    return bitwise_and_wrapper
