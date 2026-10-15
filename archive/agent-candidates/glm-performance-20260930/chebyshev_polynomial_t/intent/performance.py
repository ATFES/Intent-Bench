import torch
import intent
import intent.language as I


@intent.kernel
def chebyshev_kernel(
    input: I.In[I.f32, ("N",)],
    n: I.i32,
    out: I.Out[I.f32, ("N",)],
):
    for i in I.parallel(I.domain(0, input.shape[0])):
        value = input[i]
        previous = I.cast(1.0, I.f32)
        current = value
        for degree in I.domain(2, n + 1):
            following = 2.0 * value * current - previous
            previous = current
            current = following
        out[i] = I.select(n == 0, I.cast(1.0, I.f32), current)


def build(context):
    artifact = context.compile("chebyshev_polynomial_t", chebyshev_kernel)

    def wrapper(input, n, out=None):
        if out is None:
            out = torch.empty_like(input)
        artifact(input, n, out)
        return out

    return wrapper
