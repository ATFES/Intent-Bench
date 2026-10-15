import torch
import triton
import triton.language as tl


@triton.jit
def _sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.sqrt(values), mask=mask)


def build(context):
    def sqrt(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        _sqrt_kernel[(triton.cdiv(n_elements, 1024),)](
            input, out, n_elements, BLOCK=1024
        )
        return out

    return sqrt
