from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def add_f32_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
    alpha: I.f32,
):
    size = input.shape[0]
    idx = I.domain(0, size)
    out[idx] = input[idx] + alpha * other[idx]


def build(context):
    add_artifact = context.compile("add_f32_1d", add_f32_kernel)

    def add(input, other, *, alpha=1, out=None):
        alpha_f32 = float(alpha)
        if out is None:
            return add_artifact.run(input, other, alpha_f32)
        add_artifact(input, other, out, alpha_f32)
        return out

    return add
