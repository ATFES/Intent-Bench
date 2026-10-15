import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, y, mask=mask)


def build(context):
    def sigmoid(input, *, out=None):
        output = input if out is None else out
        if out is None:
            output = torch.empty_like(input)
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        _sigmoid_kernel[grid](input, output, n_elements, BLOCK=1024, num_warps=4)
        return output

    return sigmoid
