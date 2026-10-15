import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _asin_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # CUDA's libdevice asin is defined for fp32/fp64.  Doing the elementary
    # function in fp32 also avoids the reduced-precision half implementation.
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = libdevice.asin(values)
    tl.store(output_ptr + offsets, result.to(tl.float16), mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        _asin_kernel[grid](input, output, n_elements, BLOCK_SIZE=1024, num_warps=4)
        return output

    return wrapper
