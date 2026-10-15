import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _i0_kernel(input_ptr, output_ptr, count, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < count
    values = tl.load(input_ptr + offsets, mask=valid, other=0.0)
    results = libdevice.cyl_bessel_i0(values)
    tl.store(output_ptr + offsets, results, mask=valid)


def build(context):
    def i0(input_tensor, out=None):
        output = out if out is not None else torch.empty_like(input_tensor)
        count = input_tensor.numel()
        _i0_kernel[(triton.cdiv(count, 1024),)](
            input_tensor,
            output,
            count,
            BLOCK=1024,
        )
        return output

    return i0
