import torch
import triton
import triton.language as tl


@triton.jit
def _mul_tensor_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * y, mask=mask)


@triton.jit
def _mul_scalar_kernel(x_ptr, out_ptr, scalar, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * scalar, mask=mask)


def build(context):
    def wrapper(input, other, out=None):
        result = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 256),)
        if isinstance(other, torch.Tensor):
            _mul_tensor_kernel[grid](input, other, result, n_elements, BLOCK=256)
        else:
            _mul_scalar_kernel[grid](input, result, other, n_elements, BLOCK=256)
        return result

    return wrapper
