import torch
import triton
import triton.language as tl


@triton.jit
def _cos_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, tl.cos(values), mask=mask)


def build(context):
    del context

    def wrapper(input, *, out=None):
        if out is None:
            out = torch.empty_like(input)

        n_elements = input.numel()
        if n_elements:
            _cos_kernel[(triton.cdiv(n_elements, 1024),)](
                input,
                out,
                n_elements,
                BLOCK_SIZE=1024,
            )
        return out

    return wrapper
