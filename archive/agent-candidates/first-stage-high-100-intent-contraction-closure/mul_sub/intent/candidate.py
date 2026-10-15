import torch
import intent
import intent.language as I


@intent.kernel
def _mul_sub_kernel(
    input: I.In[I.f32, ("N",)],
    other_mul: I.In[I.f32, ("N",)],
    other_sub: I.In[I.f32, ("N",)],
    alpha: I.f32,
    out: I.Out[I.f32, ("N",)],
):
    N, = input.shape
    elements = I.domain(0, N)
    out[elements] = input[elements] * other_mul[elements] - alpha * other_sub[elements]


def build(context):
    artifact = context.compile("mul_sub_f32_1d", _mul_sub_kernel)

    def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
        if out is None:
            return artifact.run(input, other_mul, other_sub, alpha)
        artifact(input, other_mul, other_sub, alpha, out)
        return out

    return mul_sub
