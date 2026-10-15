from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 2048


@triton.jit
def _gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    IS_TANH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(input_ptr + offsets, mask=mask)

    if IS_TANH:
        inner = (value + 0.044715 * value * value * value) * 0.7978845608028654
        exp_value = tl.exp(2.0 * inner)
        tanh_value = 1.0 - 2.0 / (exp_value + 1.0)
        result = 0.5 * value * (1.0 + tanh_value)
    else:
        cdf = 0.5 * (1.0 + tl.erf(value * 0.7071067811865476))
        result = value * cdf

    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def gelu(input: torch.Tensor, approximate: str = "none") -> torch.Tensor:
        if approximate not in ("none", "tanh"):
            raise ValueError(f"approximate must be 'none' or 'tanh', got {approximate!r}")

        output = torch.empty_like(input)
        n_elements = input.numel()
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _gelu_kernel[grid](
            input,
            output,
            n_elements,
            IS_TANH=(approximate == "tanh"),
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
        return output

    return gelu
