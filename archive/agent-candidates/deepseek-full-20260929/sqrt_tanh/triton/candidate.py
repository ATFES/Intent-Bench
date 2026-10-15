import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def sqrt_tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.sqrt_rn(x)
    z = libdevice.tanh(y)
    tl.store(output_ptr + offsets, z, mask=mask)


def build(context):
    def sqrt_tanh(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        BLOCK = 1024
        grid = (triton.cdiv(n_elements, BLOCK),)
        sqrt_tanh_kernel[grid](input, out, n_elements, BLOCK=BLOCK, num_warps=8)
        return out

    return sqrt_tanh
