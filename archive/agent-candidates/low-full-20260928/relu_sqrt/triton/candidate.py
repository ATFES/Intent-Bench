import torch
import triton
import triton.language as tl


@triton.jit
def _relu_sqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Keep NaNs unchanged, matching ReLU's elementwise comparison behavior.
    values = tl.where(values < 0.0, 0.0, values)
    result = tl.sqrt(values)
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def relu_sqrt(input, inplace=False, out=None):
        if inplace:
            output = input
        elif out is not None:
            output = out
        else:
            output = torch.empty_like(input)

        n_elements = input.numel()
        _relu_sqrt_kernel[(triton.cdiv(n_elements, 256),)](
            input, output, n_elements, BLOCK=256
        )
        return output

    return relu_sqrt
