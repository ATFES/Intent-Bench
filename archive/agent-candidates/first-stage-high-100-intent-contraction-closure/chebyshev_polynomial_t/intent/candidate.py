import torch
import intent
import intent.language as I


@intent.kernel
def _chebyshev_t3(
    input: I.In[I.f32, ("M",)],
    out: I.Out[I.f32, ("M",)],
):
    M, = input.shape
    rows = I.domain(0, M)
    x = input[rows]

    one = I.cast(1.0, I.f32)
    two = I.cast(2.0, I.f32)
    t0 = I.full(x.shape, one, dtype=I.f32)
    t1 = x
    t2 = two * x * t1 - t0
    t3 = two * x * t2 - t1
    out[rows] = t3


def build(context):
    artifact = context.compile("chebyshev_t3", _chebyshev_t3)

    def chebyshev_polynomial_t(
        input: torch.Tensor,
        n: int,
        out: torch.Tensor = None,
    ) -> torch.Tensor:
        if out is None:
            return artifact.run(input)
        artifact(input, out)
        return out

    return chebyshev_polynomial_t
