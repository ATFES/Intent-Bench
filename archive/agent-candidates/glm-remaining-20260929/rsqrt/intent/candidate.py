from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def _rsqrt(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
) -> None:
    for index in I.parallel(I.domain(0, input.shape[0])):
        output[index] = I.rsqrt(input[index])


def build(context):
    rsqrt = context.compile("rsqrt_elementwise", _rsqrt)

    def rsqrt(input: "torch.Tensor", out: "torch.Tensor" = None) -> "torch.Tensor":
        if out is None:
            out = input.new_empty(input.shape)
        rsqrt(input, out)
        return out

    return rsqrt
