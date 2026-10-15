import torch
import triton
import triton.language as tl


@triton.jit
def _div_kernel(
    input_ptr,
    other_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    if NEED_MASK:
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask)
        y = tl.load(other_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x / y, mask=mask)
    else:
        x = tl.load(input_ptr + offsets)
        y = tl.load(other_ptr + offsets)
        tl.store(out_ptr + offsets, x / y)


_BLOCK_SIZE = 1024


def div(input, other, *, rounding_mode=None, out=None):
    if rounding_mode is not None:
        raise NotImplementedError("rounding_mode is not supported for this profile")

    if out is None:
        out = torch.empty_like(input)

    n_elements = input.numel()
    need_mask = (n_elements % _BLOCK_SIZE) != 0
    grid = (n_elements // _BLOCK_SIZE if not need_mask else (n_elements + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)

    _div_kernel[grid](
        input,
        other,
        out,
        n_elements,
        BLOCK_SIZE=_BLOCK_SIZE,
        NEED_MASK=need_mask,
        num_warps=4,
    )
    return out


def build(context):
    return div
