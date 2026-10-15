import torch
import triton
import triton.language as tl


BLOCK_SIZE = 1024


@triton.jit
def _sqrt_kernel(input_ptr, output_ptr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, tl.sqrt(values))


def build(context):
    def sqrt(input, out=None):
        if out is None:
            out = torch.empty_like(input)

        count = input.numel()
        _sqrt_kernel[(count // BLOCK_SIZE,)](input, out)
        return out

    return sqrt
