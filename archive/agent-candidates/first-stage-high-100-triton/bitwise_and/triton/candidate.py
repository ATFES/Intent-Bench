import torch
import triton
import triton.language as tl


@triton.jit
def _bitwise_and_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    input_values = tl.load(input_ptr + offsets, mask=mask)
    other_values = tl.load(other_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input_values & other_values, mask=mask)


def build(context):
    def wrapper(input, other, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        _bitwise_and_kernel[(triton.cdiv(n_elements, 1024),)](
            input,
            other,
            output,
            n_elements,
            BLOCK_SIZE=1024,
        )
        return output

    return wrapper
