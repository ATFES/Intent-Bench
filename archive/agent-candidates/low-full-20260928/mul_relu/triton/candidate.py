import torch
import triton
import triton.language as tl


@triton.jit
def _mul_relu_kernel(x, y, out, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x_value = tl.load(x + offsets, mask=mask)
    y_value = tl.load(y + offsets, mask=mask)
    product = x_value * y_value
    tl.store(out + offsets, tl.maximum(product, 0.0), mask=mask)


def build(context):
    def mul_relu(input, other, inplace=False, out=None):
        if inplace:
            output = input
        elif out is not None:
            output = out
        else:
            output = torch.empty_like(input)

        n_elements = input.numel()
        _mul_relu_kernel[(triton.cdiv(n_elements, 256),)](
            input, other, output, n_elements, BLOCK=256
        )
        return output

    return mul_relu
