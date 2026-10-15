import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    # Triton's sigmoid intrinsic provides the elementwise expit operation in fp32.
    y = tl.sigmoid(x)
    tl.store(out_ptr + offsets, y, mask=mask)


def build(context):
    def sigmoid(input, out=None):
        result = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        _sigmoid_kernel[(triton.cdiv(n_elements, 1024),)](
            input, result, n_elements, BLOCK=1024
        )
        return result

    return sigmoid
