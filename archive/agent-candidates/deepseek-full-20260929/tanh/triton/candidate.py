import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh as libdevice_tanh


@triton.jit
def tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)
    y = libdevice_tanh(x)
    tl.store(output_ptr + offsets, y, mask=mask)


def build(context):
    def tanh(input_tensor, out_tensor=None):
        output_tensor = (
            torch.empty_like(input_tensor)
            if out_tensor is None
            else out_tensor
        )

        n_elements = input_tensor.numel()
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        tanh_kernel[grid](
            input_tensor,
            output_tensor,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
        return output_tensor

    return tanh
