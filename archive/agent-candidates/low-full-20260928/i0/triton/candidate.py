import torch
import triton
import triton.language as tl


@triton.jit
def _i0_kernel(x_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    ax = tl.abs(x)

    # Cephes' minimax approximations are accurate over the full float32 range.
    small_y = (x / 3.75) * (x / 3.75)
    small = 1.0 + small_y * (
        3.5156229 + small_y * (
            3.0899424 + small_y * (
                1.2067492 + small_y * (
                    0.2659732 + small_y * (
                        0.360768e-1 + small_y * 0.45813e-2
                    )
                )
            )
        )
    )

    large_y = 3.75 / ax
    large_poly = (
        0.39894228 + large_y * (
            0.1328592e-1 + large_y * (
                0.225319e-2 + large_y * (
                    -0.157565e-2 + large_y * (
                        0.916281e-2 + large_y * (
                            -0.2057706e-1 + large_y * (
                                0.2635537e-1 + large_y * (
                                    -0.1647633e-1 + large_y * 0.392377e-2
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    large = tl.exp(ax) * large_poly / tl.sqrt(ax)
    result = tl.where(ax < 3.75, small, large)
    tl.store(out_ptr + offsets, result, mask=mask)


def build(context):
    def i0(input_tensor, out=None):
        output = out if out is not None else torch.empty_like(input_tensor)
        n_elements = input_tensor.numel()
        _i0_kernel[(triton.cdiv(n_elements, 256),)](
            input_tensor, output, n_elements, BLOCK=256
        )
        return output

    return i0
