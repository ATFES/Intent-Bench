from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sub_gelu_tensor_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha,
    approximate: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(other_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = x - alpha * y

    if approximate == "none":
        result = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    else:
        c = 0.7978845608028654
        result = 0.5 * x * (1.0 + tl.math.tanh(c * (x + 0.044715 * x * x * x)))

    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def _sub_gelu_scalar_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    other,
    alpha,
    approximate: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = x - alpha * other

    if approximate == "none":
        result = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
    else:
        c = 0.7978845608028654
        result = 0.5 * x * (1.0 + tl.math.tanh(c * (x + 0.044715 * x * x * x)))

    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, other, alpha=1, approximate="none", out=None):
        if out is None:
            out = torch.empty_like(input)

        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        if isinstance(other, torch.Tensor):
            _sub_gelu_tensor_kernel[grid](
                input,
                other,
                out,
                n_elements,
                float(alpha),
                approximate=approximate,
                BLOCK=1024,
                num_warps=8,
            )
        else:
            _sub_gelu_scalar_kernel[grid](
                input,
                out,
                n_elements,
                other,
                float(alpha),
                approximate=approximate,
                BLOCK=1024,
                num_warps=8,
            )
        return out

    return wrapper
