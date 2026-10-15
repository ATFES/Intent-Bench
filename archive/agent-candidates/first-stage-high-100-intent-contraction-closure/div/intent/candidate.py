import torch
import intent
import intent.language as I


@intent.kernel
def _div_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    for i in I.parallel(I.domain(0, n)):
        output[i] = I.fdiv(input[i], other[i])


def build(context):
    artifact = context.compile("div_float32", _div_kernel)

    def div(input, other, *, rounding_mode=None, out=None):
        if rounding_mode is not None:
            raise ValueError("rounding_mode must be None for this profile")
        if out is None:
            return artifact.run(input, other)
        artifact(input, other, out)
        return out

    return div
