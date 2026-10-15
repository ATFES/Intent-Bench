from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _log1p_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compensating the rounding in 1 + x is important here: for values
    # smaller than machine epsilon, the rounded sum is exactly one.  The
    # correction also retains the extra accuracy of log1p over log(1 + x)
    # throughout the neighborhood of zero.
    one_plus_x = 1.0 + x
    correction = ((one_plus_x - 1.0) - x) / one_plus_x
    result = tl.log(one_plus_x) - correction

    # The compensated expression evaluates 0/0 for x == -1 and inf-inf for
    # x == +inf.  Select the IEEE log1p values for those two special cases.
    result = tl.where(x == -1.0, -float("inf"), result)
    result = tl.where(x == float("inf"), float("inf"), result)
    tl.store(out_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        _log1p_kernel[(triton.cdiv(n_elements, 1024),)](
            input, output, n_elements, BLOCK=1024, num_warps=4
        )
        return output

    return wrapper
