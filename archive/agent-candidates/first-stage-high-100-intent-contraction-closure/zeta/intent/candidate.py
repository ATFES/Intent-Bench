from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def zeta_kernel(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    x = input[elements]
    q = other[elements]

    zero = I.cast(0.0, I.f32)
    one = I.cast(1.0, I.f32)
    half = I.cast(0.5, I.f32)
    eight = I.cast(8.0, I.f32)

    # Shift the Euler-Maclaurin tail by eight terms.  The signed finite
    # prefix also gives the recurrence needed for integer x and -8 < q <= 0.
    prefix = I.full(x.shape, 0.0, dtype=I.f32)
    even_x = I.floor(x * half) * I.cast(2.0, I.f32) == x
    for k in range(8):
        offset = I.cast(k, I.f32)
        term_arg = q + offset
        magnitude = I.exp(-x * I.log(I.abs(term_arg)))
        signed = I.select(
            term_arg < zero,
            I.select(even_x, magnitude, -magnitude),
            magnitude,
        )
        prefix = prefix + signed

    tail_arg = q + eight
    log_tail_arg = I.log(tail_arg)
    tail_power = I.exp(-x * log_tail_arg)
    tail_sq = tail_arg * tail_arg

    # Euler-Maclaurin corrections through B_12.  Keeping all arithmetic in
    # f32 matches the input/output contract and avoids an implicit widening.
    result = prefix
    result = result + I.fdiv(I.exp((one - x) * log_tail_arg), x - one)
    result = result + half * tail_power

    p1 = I.fdiv(tail_power, tail_arg)
    rising = x
    result = result + (I.cast(1.0 / 12.0, I.f32) * rising * p1)

    p3 = I.fdiv(p1, tail_sq)
    rising = rising * (x + one) * (x + I.cast(2.0, I.f32))
    result = result - (I.cast(1.0 / 720.0, I.f32) * rising * p3)

    p5 = I.fdiv(p3, tail_sq)
    rising = rising * (x + I.cast(3.0, I.f32)) * (x + I.cast(4.0, I.f32))
    result = result + (I.cast(1.0 / 30240.0, I.f32) * rising * p5)

    p7 = I.fdiv(p5, tail_sq)
    rising = rising * (x + I.cast(5.0, I.f32)) * (x + I.cast(6.0, I.f32))
    result = result - (I.cast(1.0 / 1209600.0, I.f32) * rising * p7)

    p9 = I.fdiv(p7, tail_sq)
    rising = rising * (x + I.cast(7.0, I.f32)) * (x + I.cast(8.0, I.f32))
    result = result + (I.cast(1.0 / 47900160.0, I.f32) * rising * p9)

    p11 = I.fdiv(p9, tail_sq)
    rising = rising * (x + I.cast(9.0, I.f32)) * (x + I.cast(10.0, I.f32))
    result = result - (I.cast(691.0 / 1307674368000.0, I.f32) * rising * p11)

    nan_value = I.fdiv(zero, zero)
    positive_infinity = I.cast(I.inf, I.f32)

    q_is_integer = I.floor(q) == q
    x_is_integer = I.floor(x) == x
    negative_integer_x = I.select(q > -eight, result, nan_value)
    nonpositive_q = I.select(
        q_is_integer,
        positive_infinity,
        I.select(x_is_integer, negative_integer_x, nan_value),
    )
    result = I.select(q <= zero, nonpositive_q, result)
    result = I.select(x == one, positive_infinity, result)
    result = I.select(x < one, nan_value, result)

    I.store(output, elements, result)


def build(context):
    artifact = context.compile("hurwitz_zeta_f32", zeta_kernel, constexprs={})

    def zeta(input, other, out=None):
        if out is None:
            return artifact.run(input, other)
        artifact(input, out, other)
        return out

    return zeta
