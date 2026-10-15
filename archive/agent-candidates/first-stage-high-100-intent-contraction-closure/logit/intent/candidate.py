import intent
import intent.language as I


@intent.kernel
def _logit_none(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    n = input.shape[0]
    elements = I.domain(0, n)
    x = input[elements]
    one = I.cast(1.0, I.f32)
    output[elements] = I.log(I.fdiv(x, one - x))


@intent.kernel
def _logit_eps(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
    eps: I.f32,
):
    n = input.shape[0]
    elements = I.domain(0, n)
    x = input[elements]
    one = I.cast(1.0, I.f32)
    upper = one - eps
    z = I.select(x < eps, eps, I.select(x > upper, upper, x))
    output[elements] = I.log(I.fdiv(z, one - z))


def build(context):
    unclamped = context.compile("logit_none", _logit_none, constexprs={})
    clamped = context.compile("logit_eps", _logit_eps, constexprs={})

    def logit(input, eps=None, *, out=None):
        if eps is None:
            if out is None:
                return unclamped.run(input)
            unclamped(input, out)
            return out
        if out is None:
            return clamped.run(input, eps)
        clamped(input, out, eps)
        return out

    return logit
