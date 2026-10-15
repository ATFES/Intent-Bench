import torch
import intent
import intent.language as I


@intent.kernel
def floor_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    for index in I.parallel(I.domain(0, input.shape[0])):
        output[index] = I.floor(input[index])


def build(context):
    artifact = context.compile("floor_float32", floor_kernel)

    def floor(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            return artifact.run(input)
        artifact(input, out)
        return out

    return floor
