import torch
import triton
import triton.language as tl


@triton.jit
def _rsqrt_kernel(input_ptr, output_ptr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, tl.rsqrt(values))


def build(context):
    def rsqrt(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        output = out if out is not None else torch.empty_like(input)
        block_size = 1024
        grid = (triton.cdiv(input.numel(), block_size),)
        _rsqrt_kernel[grid](input, output, BLOCK=block_size, num_warps=4)
        return output

    return rsqrt
