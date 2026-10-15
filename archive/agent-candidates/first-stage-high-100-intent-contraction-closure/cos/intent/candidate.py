import torch
import intent
import intent.language as I


@intent.kernel
def cos_kernel(
    input_tensor: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input_tensor.shape[0]
    positions = I.domain(0, n)
    output[positions] = I.cos(input_tensor[positions])


def build(context):
    artifact = context.compile("cos_kernel", cos_kernel)

    def cos(input_tensor):
        output = torch.empty_like(input_tensor)
        artifact(input_tensor, output)
        return output

    return cos
