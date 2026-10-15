from __future__ import annotations

import torch
import intent
import intent.language as I


@intent.kernel
def _logit(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
    eps: I.f32,
    CLAMPED: I.Constexpr[bool],
) -> None:
    values = input[:]

    if CLAMPED:
        one = I.cast(1.0, I.f32)
        upper = one - eps
        values = I.select(values < eps, eps, values)
        values = I.select(values > upper, upper, values)

    denominator = I.cast(1.0, I.f32) - values
    output[:] = I.log(values / denominator)


def build(context):
    unclamped = context.compile(
        "logit_unclamped", _logit, constexprs={"CLAMPED": False}
    )
    clamped = context.compile(
        "logit_clamped", _logit, constexprs={"CLAMPED": True}
    )

    def logit(input: torch.Tensor, eps: float | None = None, *, out: torch.Tensor | None = None) -> torch.Tensor:
        result = input.new_empty(input.shape) if out is None else out
        if eps is None:
            unclamped(input, result, 0.0)
        else:
            clamped(input, result, float(eps))
        return result

    return logit
