from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _sub_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
    alpha: I.f32,
):
    (n,) = input.shape
    elements = I.domain(0, n)
    output[elements] = input[elements] - alpha * other[elements]


def build(context):
    kernel = context.compile("sub_float32_1d", _sub_kernel, constexprs={})

    def sub(
        input: torch.Tensor,
        other: torch.Tensor,
        *,
        alpha: float = 1,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        if out is None:
            return kernel.run(input, other, alpha)
        kernel(input, other, out, alpha)
        return out

    return sub
