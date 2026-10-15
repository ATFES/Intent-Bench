import torch
import triton
import triton.language as tl


@triton.jit
def _leaky_relu_kernel(input_ptr, output_ptr, negative_slope, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets)
    activated = tl.where(values >= 0, values, values * negative_slope)
    tl.store(output_ptr + offsets, activated)


def build(context):
    def leaky_relu(input, negative_slope=0.01, inplace=False):
        output = input if inplace else torch.empty_like(input)
        n_elements = input.numel()
        grid = (n_elements // 1024,)
        _leaky_relu_kernel[grid](input, output, negative_slope, BLOCK_SIZE=1024, num_warps=4)
        return output

    return leaky_relu
