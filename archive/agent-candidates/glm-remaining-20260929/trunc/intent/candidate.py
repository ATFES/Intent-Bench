from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def _trunc_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
) -> None:
    values = input[:]
    zero = I.cast(0.0, I.f32)
    magnitude = I.floor(I.abs(values))
    output[:] = I.select(values >= zero, magnitude, -magnitude)


def build(context):
    trunc_kernel = context.compile("trunc_elementwise", _trunc_kernel)

    def trunc(input, out=None):
        if out is None:
            return trunc_kernel.run(input)
        trunc_kernel(input, out)
        return out

    return trunc
