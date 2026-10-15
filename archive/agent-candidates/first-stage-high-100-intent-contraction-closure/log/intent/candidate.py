from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def log_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    out[elements] = I.log(input[elements])


def build(context):
    compiled_log = context.compile("log_elementwise", log_kernel)

    def log(
        input: torch.Tensor,
        *,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        compiled_log(input, out)
        return out

    return log
