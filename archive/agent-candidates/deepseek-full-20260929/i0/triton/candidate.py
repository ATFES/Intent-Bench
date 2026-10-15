import torch
import triton
import triton.language as tl

try:
    from triton.language.extra.cuda import libdevice
    if not hasattr(libdevice, "cyl_bessel_i0"):
        raise ImportError
except Exception:
    from triton.language.extra import libdevice


@triton.jit
def _i0_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = libdevice.cyl_bessel_i0(x)
    tl.store(output_ptr + offsets, y, mask=mask)


def build(context):
    def i0(input_tensor, out=None):
        if out is None:
            out = torch.empty_like(input_tensor)
        n_elements = input_tensor.numel()
        grid = (triton.cdiv(n_elements, 1024),)
        _i0_kernel[grid](input_tensor, out, n_elements, BLOCK=1024, num_warps=4)
        return out

    return i0
