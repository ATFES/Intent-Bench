from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _add_gelu_tensor_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha,
    APPROXIMATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(other_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = x + alpha * y

    if APPROXIMATE:
        c = 0.7978845608028654
        x = 0.5 * x * (1.0 + tl.tanh(c * (x + 0.044715 * x * x * x)))
    else:
        x = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))

    tl.store(output_ptr + offsets, x, mask=mask)


@triton.jit
def _add_gelu_scalar_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    other,
    alpha,
    APPROXIMATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x = x + alpha * other

    if APPROXIMATE:
        c = 0.7978845608028654
        x = 0.5 * x * (1.0 + tl.tanh(c * (x + 0.044715 * x * x * x)))
    else:
        x = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865476))

    tl.store(output_ptr + offsets, x, mask=mask)


def build(context):
    del context

    def wrapper(input, other, alpha=1, approximate="none", out=None):
        if approximate not in ("none", "tanh"):
            raise ValueError("approximate must be 'none' or 'tanh'")

        output = out if out is not None else torch.empty_like(input)

        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        use_approximate = approximate == "tanh"

        if isinstance(other, torch.Tensor):
            _add_gelu_tensor_kernel[grid](
                input,
                other,
                output,
                n_elements,
                alpha,
                APPROXIMATE=use_approximate,
                BLOCK=1024,
            )
        else:
            _add_gelu_scalar_kernel[grid](
                input,
                output,
                n_elements,
                other,
                alpha,
                APPROXIMATE=use_approximate,
                BLOCK=1024,
            )
        return output

    return wrapper
