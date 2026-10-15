from __future__ import annotations

from typing import Tuple

import intent
import intent.language as I
import torch


@intent.kernel
def erfc_sqrt_elementwise(
    input: I.In[I.f32, ("N",)],
    erfc: I.Out[I.f32, ("N",)],
    root: I.Out[I.f32, ("N",)],
) -> None:
    N = input.shape[0]
    elements = I.domain(0, N)
    values = input[elements]
    erfc[elements] = I.erfc(values)
    root[elements] = I.sqrt(values)


def build(context):
    kernel = context.compile("erfc_sqrt_elementwise", erfc_sqrt_elementwise)

    def erfc_sqrt(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return kernel.run(input)

    return erfc_sqrt
