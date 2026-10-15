import torch
import triton
import triton.language as tl


@triton.jit
def _div_kernel(x, y, out, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    lhs = tl.load(x + offsets, mask=mask)
    rhs = tl.load(y + offsets, mask=mask)
    tl.store(out + offsets, lhs / rhs, mask=mask)


def build(context):
    def wrapper(input, other, *, rounding_mode=None, out=None):
        if rounding_mode is not None:
            raise NotImplementedError("fixed profile uses rounding_mode=None")

        result = out if out is not None else torch.empty_like(input)
        n_elements = input.numel()
        _div_kernel[(triton.cdiv(n_elements, 1024),)](
            input, other, result, n_elements, BLOCK=1024
        )
        return result

    return wrapper
