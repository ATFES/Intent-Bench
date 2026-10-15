import torch
import triton
import triton.language as tl


@triton.jit
def _abs_kernel(input_ptr, output_ptr, num_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.abs(values), mask=mask)


def build(context):
    del context

    def abs(input_tensor, out=None):
        output = torch.empty_like(input_tensor) if out is None else out
        num_elements = input_tensor.numel()
        _abs_kernel[(triton.cdiv(num_elements, 1024),)](
            input_tensor,
            output,
            num_elements,
            BLOCK_SIZE=1024,
        )
        return output

    return abs
