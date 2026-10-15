from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _log1p(
    values: I.In[I.f32, ("N",)],
    results: I.Out[I.f32, ("N",)],
) -> None:
    for index in I.parallel(I.domain(0, values.shape[0])):
        results[index] = I.log1p(values[index])


def build(context):
    kernel = context.compile("log1p_elementwise", _log1p)

    def log1p(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        result = out
        if result is None:
            result = torch.empty_like(input)
        kernel(input, result)
        return result

    return log1p
