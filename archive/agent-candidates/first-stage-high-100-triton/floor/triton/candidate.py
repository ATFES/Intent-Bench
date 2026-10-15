import torch
import triton
import triton.language as tl


@triton.jit
def _floor_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, tl.floor(values), mask=mask)


@triton.jit
def _copy_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, values, mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        result = out if out is not None else torch.empty_like(input)
        n_elements = input.numel()
        if n_elements:
            grid = (triton.cdiv(n_elements, 1024),)
            if input.dtype.is_floating_point:
                _floor_kernel[grid](input, result, n_elements, BLOCK=1024, num_warps=4)
            else:
                _copy_kernel[grid](input, result, n_elements, BLOCK=1024, num_warps=4)
        return result

    return wrapper
