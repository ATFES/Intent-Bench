import torch
import triton
import triton.language as tl


@triton.jit
def _tanh_kernel(input_ptr, output_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    values = tl.load(input_ptr + offsets)
    tl.store(output_ptr + offsets, tl.extra.cuda.libdevice.tanh(values))


def build(context):
    del context

    def tanh(input_tensor, out_tensor=None):
        if out_tensor is None:
            output_tensor = torch.empty_like(input_tensor)
        else:
            output_tensor = out_tensor

        num_elements = input_tensor.numel()
        _tanh_kernel[(num_elements // 1024,)](
            input_tensor,
            output_tensor,
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        return output_tensor

    return tanh
