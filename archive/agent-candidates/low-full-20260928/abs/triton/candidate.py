import torch
import triton
import triton.language as tl


@triton.jit
def _abs_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.abs(values), mask=mask)


def build(context):
    def wrapper(input_tensor, out=None):
        if out is None:
            out = torch.empty_like(input_tensor)
        n_elements = input_tensor.numel()
        _abs_kernel[(triton.cdiv(n_elements, 1024),)](
            input_tensor, out, n_elements, BLOCK=1024
        )
        return out

    return wrapper
