from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gammaln_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.abs(x)

    # Lanczos approximation for log(Gamma(z)) on z >= 1/2.  For the small
    # interval, Gamma(x) = Gamma(x + 1) / x avoids the sine and reflection
    # calculation while remaining well-conditioned away from the pole.
    small = x < 0.5
    y = tl.where(small, x + 1.0, x)
    z = y - 1.0
    a = 0.9999999999998099
    a = a + 676.5203681218851 / (z + 1.0)
    a = a - 1259.139216722403 / (z + 2.0)
    a = a + 771.3234287776531 / (z + 3.0)
    a = a - 176.6150291621406 / (z + 4.0)
    a = a + 12.50734327868691 / (z + 5.0)
    a = a - 0.1385710952657201 / (z + 6.0)
    a = a + 0.000009984369578019572 / (z + 7.0)
    a = a + 0.0000001505632735149312 / (z + 8.0)
    t = z + 7.5
    lanczos = 0.9189385332046727 + (z + 0.5) * tl.log(t) - t + tl.log(a)
    result = lanczos - tl.where(small, tl.log(x), 0.0)

    # Keep the special values well-defined despite the indeterminate inf-inf
    # form in the Lanczos expression for an infinite input.  NaNs pass
    # through via the final selection.
    result = tl.where(x > 1.0e37, float("inf"), result)
    result = tl.where(x == x, result, x)
    tl.store(out_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        _gammaln_kernel[(triton.cdiv(n_elements, 256),)](
            input,
            output,
            n_elements,
            BLOCK=256,
            num_warps=4,
        )
        return output

    return wrapper
