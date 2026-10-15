import torch
import intent
import intent.language as I
from typing import Tuple


@intent.kernel
def _erfc_sqrt_kernel(
    input: I.In[I.f32, ("N",)],
    erfc_out: I.Out[I.f32, ("N",)],
    sqrt_out: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    values = input[elements]
    erfc_out[elements] = I.cast(1.0, I.f32) - I.erf(values)
    sqrt_out[elements] = I.sqrt(values)


def build(context):
    kernel = context.compile(
        "erfc_sqrt_elementwise", _erfc_sqrt_kernel, constexprs={}
    )

    def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return kernel.run(input)

    return erfc_sqrt
