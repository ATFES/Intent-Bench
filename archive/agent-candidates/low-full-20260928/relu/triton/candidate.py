import torch
import triton
import triton.language as tl


@triton.jit
def _relu_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, y, mask=mask)


def build(context):
    def relu(input, inplace=False):
        output = input if inplace else torch.empty_like(input)
        n_elements = input.numel()
        _relu_kernel[(triton.cdiv(n_elements, 1024),)](
            input, output, n_elements, BLOCK=1024
        )
        return output

    return relu
