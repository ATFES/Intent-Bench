from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _selu_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    alpha = 1.6732632423543772
    scale = 1.0507009873554805
    positive = tl.maximum(x, 0.0)
    negative = tl.minimum(x, alpha * (tl.exp(x) - 1.0))
    y = scale * (positive + negative)
    tl.store(y_ptr + offsets, y, mask=mask)


def build(context):
    def selu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        output = torch.empty_like(input)
        n_elements = input.numel()
        _selu_kernel[(triton.cdiv(n_elements, 1024),)](
            input, output, n_elements, BLOCK=1024, num_warps=4
        )
        return output

    return selu
