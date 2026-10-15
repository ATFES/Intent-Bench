import intent
import intent.language as I


@intent.kernel
def mul_kernel(
    input: I.In[I.f16, (1048576,)],
    other: I.In[I.f16, (1048576,)],
    out: I.Out[I.f16, (1048576,)],
):
    elements = I.domain(0, 1048576)
    out[elements] = input[elements] * other[elements]


def build(context):
    artifact = context.compile("mul_f16_1048576", mul_kernel)

    def mul(input, other, out=None):
        if out is None:
            return artifact.run(input, other)
        artifact(input, other, out)
        return out

    return mul
