import torch
import triton
import triton.language as tl


@triton.jit
def _leaky_relu_kernel(x_ptr, out_ptr, n_elements, negative_slope, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    result = tl.where(x < 0, x * negative_slope, x)
    tl.store(out_ptr + offsets, result, mask=mask)


def build(context):
    def leaky_relu(input, negative_slope=0.01, inplace=False):
        output = input if inplace else torch.empty_like(input)
        n_elements = input.numel()
        _leaky_relu_kernel[(triton.cdiv(n_elements, 1024),)](
            input,
            output,
            n_elements,
            negative_slope,
            BLOCK=1024,
        )
        return output

    return leaky_relu
