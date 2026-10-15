from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _log_tanh(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
) -> None:
    for element_index in I.parallel(I.domain(0, input.shape[0])):
        log_value = I.log(input[element_index])
        out[element_index] = I.tanh(log_value)


def build(context):
    kernel = context.compile("log_tanh_elementwise", _log_tanh)

    def log_tanh(input, out=None):
        result = out if out is not None else torch.empty(
            input.shape,
            dtype=input.dtype,
            device=input.device,
        )
        kernel(input, result)
        return result

    return log_tanh
