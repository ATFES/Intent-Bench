import torch
import intent
import intent.language as I


@intent.kernel
def cos_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output_tensor: I.Out[I.f32, ("N",)],
):
    elements = I.domain(0, input_tensor.shape[0])
    for index in I.parallel(elements):
        output_tensor[index] = I.cos(input_tensor[index])


def build(context):
    artifact = context.compile("cos_pointwise", cos_kernel)

    def cos(input_tensor):
        return artifact.run(input_tensor)

    return cos
