import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _asin_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    result = libdevice.asin(values)
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def asin(input_tensor):
        output = torch.empty_like(input_tensor)
        n_elements = input_tensor.numel()
        _asin_kernel[(triton.cdiv(n_elements, 1024),)](
            input_tensor, output, n_elements, BLOCK=1024, num_warps=4
        )
        return output

    return asin
