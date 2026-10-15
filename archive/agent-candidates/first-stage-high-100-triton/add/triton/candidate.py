import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(input_ptr, other_ptr, output_ptr, n_elements, alpha, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    input_value = tl.load(input_ptr + offsets, mask=mask)
    other_value = tl.load(other_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input_value + alpha * other_value, mask=mask)


def build(context):
    def wrapper(input, other, *, alpha=1, out=None):
        output = out if out is not None else torch.empty_like(input)
        n_elements = input.numel()
        if n_elements:
            _add_kernel[(triton.cdiv(n_elements, 1024),)](
                input,
                other,
                output,
                n_elements,
                alpha,
                BLOCK=1024,
                num_warps=4,
            )
        return output

    return wrapper
