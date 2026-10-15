import torch
import triton
import triton.language as tl


@triton.jit
def _cos_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, tl.cos(values), mask=mask)


def build(context):
    def cos(input_tensor):
        output = torch.empty_like(input_tensor)
        n_elements = input_tensor.numel()
        _cos_kernel[(triton.cdiv(n_elements, 1024),)](
            input_tensor, output, n_elements, BLOCK=1024
        )
        return output

    return cos
