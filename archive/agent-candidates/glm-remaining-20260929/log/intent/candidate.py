from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _natural_log(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
) -> None:
    elements = I.domain(0, input.shape[0])
    output[elements] = I.log(input[elements])


def build(context):
    log_kernel = context.compile("natural_log", _natural_log)

    def log(input: "torch.Tensor", out: "torch.Tensor" = None) -> "torch.Tensor":
        output = out
        if output is None:
            output = torch.empty_like(input)
        log_kernel(input, output)
        return output

    return log
