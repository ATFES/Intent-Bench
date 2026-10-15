import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_exact_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    inv_sqrt_2 = 0.7071067811865476
    result = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt_2))
    tl.store(out_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, approximate='none'):
        output = torch.empty_like(input)
        n_elements = input.numel()
        _gelu_exact_kernel[(triton.cdiv(n_elements, 256),)](
            input, output, n_elements, BLOCK_SIZE=256
        )
        return output

    return wrapper
