import torch
import triton
import triton.language as tl


ALPHA = 1.6732632423543772
SCALE = 1.0507009873554805


@triton.jit
def _selu_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    value = tl.load(input_ptr + offsets, mask=mask)

    negative = ALPHA * (tl.exp(value) - 1.0)
    selected = tl.where(value < 0.0, negative, value)
    tl.store(output_ptr + offsets, SCALE * selected, mask=mask)


def build(context):
    def selu(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
        output = torch.empty_like(input)
        n_elements = input.numel()
        block_size = 1024
        grid = ((n_elements + block_size - 1) // block_size,)
        _selu_kernel[grid](input, output, n_elements, BLOCK=block_size)
        return output

    return selu
