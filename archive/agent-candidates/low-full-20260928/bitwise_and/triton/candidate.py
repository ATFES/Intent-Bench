import torch
import triton
import triton.language as tl


@triton.jit
def _bitwise_and_kernel(input_ptr, other_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_values = tl.load(input_ptr + offsets, mask=mask)
    other_values = tl.load(other_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input_values & other_values, mask=mask)


def build(context):
    def bitwise_and(input, other, out=None):
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        _bitwise_and_kernel[(triton.cdiv(n_elements, 256),)](
            input, other, out, n_elements, BLOCK=256
        )
        return out

    return bitwise_and
