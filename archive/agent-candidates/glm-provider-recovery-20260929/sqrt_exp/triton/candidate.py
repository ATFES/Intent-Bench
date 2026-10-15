import torch
import triton
import triton.language as tl


@triton.jit
def sqrt_exp_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    value = tl.load(input_ptr + offsets, mask=mask)
    result = tl.exp(tl.sqrt_rn(value))
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def sqrt_exp(input, out=None):
        output = input if out is not None else torch.empty_like(input)
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        sqrt_exp_kernel[grid](
            input,
            output,
            n_elements,
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        return output

    return sqrt_exp
