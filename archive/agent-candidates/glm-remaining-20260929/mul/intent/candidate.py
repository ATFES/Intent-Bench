import intent
import intent.language as I


@intent.kernel
def mul_half_kernel(
    input: I.In[I.f16, ("N",)],
    other: I.In[I.f16, ("N",)],
    output: I.Out[I.f16, ("N",)],
):
    for element in I.parallel(I.domain(0, input.shape[0])):
        output[element] = input[element] * other[element]


def build(context):
    mul_half = context.compile("mul_half_elementwise", mul_half_kernel)

    def mul(input, other, out=None):
        if out is None:
            return mul_half.run(input, other)
        mul_half(input, other, out)
        return out

    return mul
