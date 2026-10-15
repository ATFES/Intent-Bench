import torch
import triton
import triton.language as tl


@triton.jit
def _i0_kernel(input_ptr, output_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    ax = tl.abs(x)

    # Power series / Chebyshev-style fits used by the standard scalar I0
    # implementation.  The two fits overlap at 3.75 and are accurate in
    # float32 over their respective ranges.
    y_small = (ax / 3.75) * (ax / 3.75)
    small = 1.0 + y_small * (
        3.5156229 + y_small * (
            3.0899424 + y_small * (
                1.2067492 + y_small * (
                    0.2659732 + y_small * (0.0360768 + y_small * 0.0045813)
                )
            )
        )
    )

    y_large = 3.75 / ax
    large_poly = 0.00392377
    large_poly = -0.01647633 + y_large * large_poly
    large_poly = 0.02635537 + y_large * large_poly
    large_poly = -0.02057706 + y_large * large_poly
    large_poly = 0.00916281 + y_large * large_poly
    large_poly = -0.00157565 + y_large * large_poly
    large_poly = 0.00225319 + y_large * large_poly
    large_poly = 0.01328592 + y_large * large_poly
    large_poly = 0.39894228 + y_large * large_poly

    # Fold the sqrt into the exponent so values around x=90 do not overflow
    # in the intermediate exp(x), even though I0(x) is still finite there.
    large = tl.exp(ax - 0.5 * tl.log(ax)) * large_poly
    result = tl.where(ax < 3.75, small, large)

    # exp(inf) - log(inf) produces inf - inf; I0(+-inf) is +inf.
    # This comparison detects infinity without requiring a special constant.
    is_inf = (ax == ax + ax) & (ax > 0.0)
    result = tl.where(is_inf, ax, result)
    tl.store(output_ptr + offsets, result, mask=mask)


def build(context):
    def wrapper(input, *, out=None):
        output = torch.empty_like(input) if out is None else out
        n_elements = input.numel()
        if n_elements == 0:
            return output
        _i0_kernel[(triton.cdiv(n_elements, 1024),)](
            input, output, n_elements, BLOCK=1024
        )
        return output

    return wrapper
