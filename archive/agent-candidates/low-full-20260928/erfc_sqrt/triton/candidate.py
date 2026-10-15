import torch
import triton
import triton.language as tl


@triton.jit
def _erfc_sqrt_kernel(x_ptr, erfc_ptr, sqrt_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(erfc_ptr + offsets, tl.erfc(x), mask=mask)
    tl.store(sqrt_ptr + offsets, tl.sqrt(x), mask=mask)


def build(context):
    def erfc_sqrt(input: torch.Tensor):
        erfc_out = torch.empty_like(input)
        sqrt_out = torch.empty_like(input)
        n_elements = input.numel()
        _erfc_sqrt_kernel[(triton.cdiv(n_elements, 256),)](
            input, erfc_out, sqrt_out, n_elements, BLOCK=256
        )
        return erfc_out, sqrt_out

    return erfc_sqrt
