from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def _tensordot_rsqrt_f32(
    a: I.In[I.f32, ("M", "K")],
    b: I.In[I.f32, ("K", "N")],
    output: I.Out[I.f32, ("M", "N")],
) -> None:
    product = I.matmul(a[:, :], b[:, :], acc_dtype=I.f32)
    output[:, :] = I.rsqrt(product)


def build(context):
    compiled_tensordot_rsqrt = context.compile(
        "_tensordot_rsqrt_f32", _tensordot_rsqrt_f32
    )

    def tensordot_rsqrt(
        a: "torch.Tensor",
        b: "torch.Tensor",
        dims: int | tuple[list[int], list[int]] | list[list[int]] = 1,
    ) -> "torch.Tensor":
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError("this profile requires two-dimensional inputs")
        if not isinstance(dims, int) or dims != 1:
            raise ValueError("this profile requires dims=1")
        if a.shape[1] != b.shape[0]:
            raise ValueError("contraction dimensions must have the same extent")

        return compiled_tensordot_rsqrt.run(a, b)

    return tensordot_rsqrt
