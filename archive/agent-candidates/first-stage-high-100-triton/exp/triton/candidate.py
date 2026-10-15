import torch
import triton
import triton.language as tl


@triton.jit
def _exp_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, tl.exp(x), mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        output = out
        if output is None:
            output = torch.empty_like(input)

        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        _exp_kernel[grid](input, output, n_elements, BLOCK=1024, num_warps=4)
        return output

    return wrapper
