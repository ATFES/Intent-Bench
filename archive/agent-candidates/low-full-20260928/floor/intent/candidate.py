import torch
import intent
import intent.language as I


@intent.kernel
def _floor_f32(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    output[elements] = I.floor(input[elements])


def build(context):
    floor_kernel = context.compile("floor_f32", _floor_f32)

    def floor(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        floor_kernel(input, out)
        return out

    return floor
