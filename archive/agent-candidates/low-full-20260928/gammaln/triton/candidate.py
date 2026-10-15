import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _gammaln_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = libdevice.lgamma(x)
    tl.store(output_ptr + offsets, y, mask=mask)


def build(context):
    def gammaln(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(input)
        n_elements = input.numel()
        grid = (triton.cdiv(n_elements, 256),)
        _gammaln_kernel[grid](input, out, n_elements, BLOCK=256)
        return out

    return gammaln
