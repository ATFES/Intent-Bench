import torch
import intent
import intent.language as I


@intent.kernel
def _i0_kernel(
    input_tensor: I.In[I.f32, ("M",)],
    output: I.Out[I.f32, ("M",)],
):
    M = input_tensor.shape[0]
    points = I.domain(0, M)
    x = input_tensor[points]
    ax = I.abs(x)

    c1 = I.cast(1.0, I.f32)
    c375 = I.cast(3.75, I.f32)

    # Cephes/Numerical Recipes piecewise approximation for I0.
    small_y = (ax / c375) * (ax / c375)
    small = c1 + small_y * (
        I.cast(3.5156229, I.f32)
        + small_y
        * (
            I.cast(3.0899424, I.f32)
            + small_y
            * (
                I.cast(1.2067492, I.f32)
                + small_y
                * (
                    I.cast(0.2659732, I.f32)
                    + small_y
                    * (
                        I.cast(0.0360768, I.f32)
                        + small_y * I.cast(0.0045813, I.f32)
                    )
                )
            )
        )
    )

    large_y = c375 / ax
    large_poly = I.cast(0.39894228, I.f32) + large_y * (
        I.cast(0.01328592, I.f32)
        + large_y
        * (
            I.cast(0.00225319, I.f32)
            + large_y
            * (
                I.cast(-0.00157565, I.f32)
                + large_y
                * (
                    I.cast(0.00916281, I.f32)
                    + large_y
                    * (
                        I.cast(-0.02057706, I.f32)
                        + large_y
                        * (
                            I.cast(0.02635537, I.f32)
                            + large_y
                            * (
                                I.cast(-0.01647633, I.f32)
                                + large_y * I.cast(0.00392377, I.f32)
                            )
                        )
                    )
                )
            )
        )
    )
    large_raw = (I.exp(ax) / I.sqrt(ax)) * large_poly
    inf32 = I.cast(I.inf, I.f32)
    large = I.select(ax == inf32, inf32, large_raw)
    result = I.select(ax < c375, small, large)
    output[points] = result


def build(context):
    artifact = context.compile("i0_kernel", _i0_kernel)

    def i0(input_tensor, out=None):
        if out is None:
            out = torch.empty_like(input_tensor)
        artifact(input_tensor, out)
        return out

    return i0
