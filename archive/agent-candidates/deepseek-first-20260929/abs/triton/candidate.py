import torch
import triton
import triton.language as tl


@triton.jit
def _abs_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.abs(x), mask=mask)


def build(context):
    def abs(input_tensor, out=None):
        output = torch.empty_like(input_tensor) if out is None else out
        n_elements = input_tensor.numel()
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _abs_kernel[grid](input_tensor, output, n_elements, BLOCK_SIZE=block_size, num_warps=4)
        return output

    return abs
