import torch
import intent
import intent.language as I


@intent.kernel
def _chebyshev(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
    n: I.i64,
):
    values = I.domain(0, input.shape[0])
    for i in I.parallel(values):
        x = input[i]
        t0 = I.cast(1.0, I.f32)
        t1 = x
        if n == 0:
            result = t0
        elif n == 1:
            result = t1
        else:
            prev = t0
            curr = t1
            degree = 1
            while degree < n:
                nxt = 2.0 * x * curr - prev
                prev = curr
                curr = nxt
                degree = degree + 1
            result = curr
        out[i] = result


def build(context):
    kernel = context.compile("chebyshev_polynomial_t", _chebyshev)

    def chebyshev_polynomial_t(input: "torch.Tensor", n: int, out: "torch.Tensor" = None) -> "torch.Tensor":
        if out is None:
            out = torch.empty_like(input)
        kernel(input, out, n)
        return out

    return chebyshev_polynomial_t
