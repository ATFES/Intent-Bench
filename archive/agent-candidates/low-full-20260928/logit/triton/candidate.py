import torch
import triton
import triton.language as tl


@triton.jit
def _logit_unclamped(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.log(x / (1.0 - x))
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def _logit_clamped(x_ptr, out_ptr, n_elements, eps, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    z = tl.maximum(tl.minimum(x, 1.0 - eps), eps)
    y = tl.log(z / (1.0 - z))
    tl.store(out_ptr + offsets, y, mask=mask)


def build(context):
    def wrapper(input, eps=None, out=None):
        result = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 256),)
        if eps is None:
            _logit_unclamped[grid](input, result, n_elements, BLOCK=256)
        else:
            _logit_clamped[grid](input, result, n_elements, eps, BLOCK=256)
        return result

    return wrapper
