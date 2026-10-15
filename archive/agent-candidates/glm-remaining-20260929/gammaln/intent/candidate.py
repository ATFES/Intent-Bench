import intent
import intent.language as I


@intent.kernel
def _gammaln_elementwise(
    input: I.In[I.f32, ("N",)],
    output: I.Out[I.f32, ("N",)],
):
    for index in I.parallel(I.domain(0, input.shape[0])):
        output[index] = I.lgamma(input[index])


def build(context):
    gammaln_elementwise = context.compile(
        "gammaln_elementwise", _gammaln_elementwise
    )

    def gammaln(input: "torch.Tensor", out: "torch.Tensor" = None):
        if out is None:
            return gammaln_elementwise.run(input)
        gammaln_elementwise(input, out)
        return out

    return gammaln
