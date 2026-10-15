import torch
import triton
import triton.language as tl


_BLOCK_SIZE = 2048
_NUM_WARPS = 8


@triton.jit
def _sqrt_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.sqrt_rn(x)
    tl.store(out_ptr + offsets, y, mask=mask)


def build(context):
    def sqrt(input, out=None):
        if out is None:
            out = torch.empty_like(input)

        n_elements = input.numel()
        if n_elements == 0:
            return out

        grid = (triton.cdiv(n_elements, _BLOCK_SIZE),)
        _sqrt_kernel[grid](
            input,
            out,
            n_elements,
            BLOCK=_BLOCK_SIZE,
            num_warps=_NUM_WARPS,
        )
        return out

    return sqrt
