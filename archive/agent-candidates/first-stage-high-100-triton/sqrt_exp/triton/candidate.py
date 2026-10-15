import torch
import triton
import triton.language as tl


@triton.jit
def _sqrt_exp_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    result = tl.exp(tl.sqrt(values))
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, out=None):
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        _sqrt_exp_kernel[(triton.cdiv(n_elements, 1024),)](
            input, out, n_elements, BLOCK=1024
        )
        return out

    return wrapper
