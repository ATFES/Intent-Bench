import torch
import triton
import triton.language as tl


@triton.jit
def _multiply_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    input_value = tl.load(input_ptr + offsets)
    other_value = tl.load(other_ptr + offsets)
    tl.store(output_ptr + offsets, input_value * other_value)


def build(context):
    def mul(input, other, out=None):
        if out is None:
            out = torch.empty_like(input)

        count = input.numel()
        block_size = 2048
        _multiply_kernel[(count // block_size,)](
            input,
            other,
            out,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        return out

    return mul
