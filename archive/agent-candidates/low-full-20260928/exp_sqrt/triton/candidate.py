import torch
import triton
import triton.language as tl


@triton.jit
def _exp_sqrt_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.sqrt(tl.exp(x))
    tl.store(out_ptr + offsets, y, mask=mask)


def build(context):
    def exp_sqrt(input, out=None):
        result = out if out is not None else torch.empty_like(input)
        n_elements = input.numel()
        _exp_sqrt_kernel[(triton.cdiv(n_elements, 1024),)](
            input, result, n_elements, BLOCK=1024
        )
        return result

    return exp_sqrt
