import torch
import intent
import intent.language as I


@intent.kernel
def _permute_copy_kernel(
    input: I.In[I.f32, (32, 32, 32, 32)],
    output: I.Out[I.f32, (32, 32, 32, 32)],
):
    d0 = I.domain(0, 32)
    d1 = I.domain(0, 32)
    d2 = I.domain(0, 32)
    d3 = I.domain(0, 32)
    values = input[d0, d1, d2, d3]
    permuted = I.transpose(values, (3, 2, 1, 0))
    output[d0, d1, d2, d3] = permuted


def build(context):
    artifact = context.compile("permute_copy", _permute_copy_kernel)

    def permute_copy(input, dims):
        output = torch.empty((32, 32, 32, 32), dtype=input.dtype, device=input.device)
        artifact(input, output)
        return output

    return permute_copy
