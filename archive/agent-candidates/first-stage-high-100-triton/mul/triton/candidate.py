import torch
import triton
import triton.language as tl


@triton.jit
def _mul_kernel(input_ptr, other_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_value = tl.load(input_ptr + offsets, mask=mask)
    other_value = tl.load(other_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input_value * other_value, mask=mask)


def build(context):
    def wrapper(input, other, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        _mul_kernel[(triton.cdiv(n_elements, 1024),)](
            input,
            other,
            output,
            n_elements,
            BLOCK=1024,
        )
        return output

    return wrapper
