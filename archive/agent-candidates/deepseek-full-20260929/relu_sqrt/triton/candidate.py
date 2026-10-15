import torch
import triton
import triton.language as tl


@triton.jit
def _relu_sqrt_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offsets)
    y = tl.sqrt(tl.maximum(x, 0.0))
    tl.store(y_ptr + offsets, y)


@triton.jit
def _relu_sqrt_kernel_masked(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.sqrt(tl.maximum(x, 0.0))
    tl.store(y_ptr + offsets, y, mask=mask)


def build(context):
    def relu_sqrt(input, inplace=False, out=None):
        n = input.numel()

        if inplace:
            dst = input
        elif out is not None:
            dst = out
        else:
            dst = torch.empty_like(input)

        if n == 0:
            return dst

        block = 1024
        if n % block == 0:
            _relu_sqrt_kernel[(n // block,)](input, dst, BLOCK=block)
        else:
            _relu_sqrt_kernel_masked[(triton.cdiv(n, block),)](input, dst, n, BLOCK=block)

        return dst

    return relu_sqrt
