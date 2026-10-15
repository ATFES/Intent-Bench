import intent
import intent.language as I


@intent.kernel
def i0_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output_tensor: I.Out[I.f32, ("N",)],
):
    for i in I.parallel(I.domain(0, input_tensor.shape[0])):
        output_tensor[i] = I.i0(input_tensor[i])


def build(context):
    artifact = context.compile("i0_kernel", i0_kernel)

    def i0(input_tensor, out=None):
        if out is None:
            return artifact.run(input_tensor)
        artifact(input_tensor, out)
        return out

    return i0
