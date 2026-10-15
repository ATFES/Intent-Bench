import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(input_ptr, other_ptr, output_ptr, n_elements, alpha, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + alpha * y, mask=mask)


def build(context):
    def add(input, other, alpha=1, out=None):
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        _add_kernel[(triton.cdiv(n_elements, 256),)](
            input, other, out, n_elements, alpha, BLOCK=256
        )
        return out

    return add
