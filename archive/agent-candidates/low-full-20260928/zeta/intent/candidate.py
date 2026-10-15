import torch
import intent
import intent.language as I


@intent.kernel
def zeta_kernel(
    x: I.In[I.f32, ("N",)],
    q: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N",)],
):
    n = x.shape[0]
    for i in I.parallel(I.domain(0, n)):
        xv = x[i]
        qv = q[i]
        # A short direct sum followed by the Euler-Maclaurin tail gives a
        # uniform approximation while keeping each element independent.
        acc = I.cast(0.0, I.f32)
        for k in range(32):
            kv = I.cast(k, I.f32)
            t = qv + kv
            acc = acc + I.exp(-xv * I.log(t))
        a = qv + I.cast(32.0, I.f32)
        amx = I.exp((I.cast(1.0, I.f32) - xv) * I.log(a))
        tail = amx / (xv - I.cast(1.0, I.f32))
        tail = tail + I.cast(0.5, I.f32) * I.exp(-xv * I.log(a))
        tail = tail + (xv / I.cast(12.0, I.f32)) * I.exp(-(xv + I.cast(1.0, I.f32)) * I.log(a))
        tail = tail - (xv * (xv + I.cast(1.0, I.f32)) * (xv + I.cast(2.0, I.f32)) / I.cast(720.0, I.f32)) * I.exp(-(xv + I.cast(3.0, I.f32)) * I.log(a))
        regular = acc + tail
        nanv = I.cast(0.0, I.f32) / I.cast(0.0, I.f32)
        neg = I.cast(-I.inf, I.f32)
        is_integer_q = I.floor(qv) == qv
        bad_nonpositive = (qv <= I.cast(0.0, I.f32))
        special = I.select(xv == I.cast(1.0, I.f32), I.inf, regular)
        special = I.select(xv < I.cast(1.0, I.f32), nanv, special)
        special = I.select(bad_nonpositive & is_integer_q & (xv > I.cast(1.0, I.f32)), I.inf, special)
        special = I.select(bad_nonpositive & (is_integer_q == False) & (xv > I.cast(1.0, I.f32)), nanv, special)
        out[i] = special


def build(context):
    compiled = context.compile("zeta_elementwise", zeta_kernel)

    def zeta(input: torch.Tensor, other: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        result = out if out is not None else torch.empty_like(input)
        compiled(input, other, result)
        return result

    return zeta
