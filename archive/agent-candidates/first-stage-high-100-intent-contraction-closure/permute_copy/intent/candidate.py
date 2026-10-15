import torch
import intent
import intent.language as I


@intent.kernel
def permute_copy_reverse(
    input: I.In[I.f32, (32, 32, 32, 32)],
    output: I.Out[I.f32, (32, 32, 32, 32)],
):
    axis0 = I.domain(0, 32)
    axis1 = I.domain(0, 32)
    axis2 = I.domain(0, 32)
    axis3 = I.domain(0, 32)

    value = input[axis0, axis1, axis2, axis3]
    output[axis0, axis1, axis2, axis3] = I.transpose(value, (3, 2, 1, 0))


def build(context):
    artifact = context.compile("permute_copy_reverse", permute_copy_reverse)

    def permute_copy(input: torch.Tensor, dims: list) -> torch.Tensor:
        return artifact.run(input)

    return permute_copy
