from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def _rsqrt_elementwise(
    inp: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
) -> None:
    for i in I.parallel(I.domain(0, inp.shape[0])):
        out[i] = I.rsqrt(inp[i])


def build(context):
    artifact = context.compile("rsqrt_elementwise", _rsqrt_elementwise)

    def rsqrt(input, *, out=None):
        if out is None:
            return artifact.run(input)
        artifact(input, out)
        return out

    return rsqrt
