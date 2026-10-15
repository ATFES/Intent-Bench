import torch
import intent
import intent.language as I


@intent.kernel
def add_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
    alpha: I.f32,
):
    n, = input.shape
    elements = I.domain(0, n)
    out[elements] = input[elements] + alpha * other[elements]


def build(context):
    artifact = context.compile("elementwise_add", add_kernel, constexprs={})

    def add(input, other, alpha=1, out=None):
        if out is None:
            return artifact.run(input, other, alpha)
        artifact(input, other, out, alpha)
        return out

    return add
