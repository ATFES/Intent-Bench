import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(input_ptr, other_ptr, out_ptr, n_elements, EVEN: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        lhs = tl.load(input_ptr + offsets)
        rhs = tl.load(other_ptr + offsets)
        tl.store(out_ptr + offsets, lhs + rhs)
    else:
        mask = offsets < n_elements
        lhs = tl.load(input_ptr + offsets, mask=mask)
        rhs = tl.load(other_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, lhs + rhs, mask=mask)


@triton.jit
def _add_alpha_kernel(input_ptr, other_ptr, out_ptr, n_elements, alpha, EVEN: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    if EVEN:
        lhs = tl.load(input_ptr + offsets)
        rhs = tl.load(other_ptr + offsets)
        tl.store(out_ptr + offsets, lhs + rhs * alpha)
    else:
        mask = offsets < n_elements
        lhs = tl.load(input_ptr + offsets, mask=mask)
        rhs = tl.load(other_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, lhs + rhs * alpha, mask=mask)


_BLOCK = 1024


def build(context):
    def add(input, other, alpha=1, out=None):
        if out is None:
            out = torch.empty_like(input)

        count = out.numel()
        even = count % _BLOCK == 0
        grid = (triton.cdiv(count, _BLOCK),)

        if alpha == 1:
            _add_kernel[grid](
                input, other, out, count,
                EVEN=even, BLOCK=_BLOCK,
            )
        else:
            _add_alpha_kernel[grid](
                input, other, out, count, alpha,
                EVEN=even, BLOCK=_BLOCK,
            )
        return out

    return add
