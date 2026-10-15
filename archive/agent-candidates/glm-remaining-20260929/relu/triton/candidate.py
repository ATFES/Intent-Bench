import torch
import triton
import triton.language as tl


@triton.jit
def _relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    result = tl.where(values > 0.0, values, 0.0)
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def relu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        output = input if inplace else torch.empty_like(input)
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 4096),)
        _relu_kernel[grid](input, output, n_elements, BLOCK_SIZE=4096)
        return output

    return relu
