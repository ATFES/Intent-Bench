import torch
import triton
import triton.language as tl


@triton.jit
def _relu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, tl.maximum(x, 0.0), mask=mask)


def build(context):
    block = 1024

    def wrapper(input, inplace=False):
        output = input if inplace else torch.empty_like(input)
        n_elements = input.numel()
        _relu_kernel[(triton.cdiv(n_elements, block),)](
            input, output, n_elements, BLOCK=block, num_warps=4
        )
        return output

    return wrapper
