import intent
import intent.language as I


@intent.kernel
def _sigmoid_argmax_kernel(
    input: I.In[I.f32, ("M", "N")],
    output: I.Out[I.i64, ("M",)],
):
    rows = I.domain(0, input.shape[0])
    sigmoid = I.sigmoid(input)
    _, indices = I.arg_reduce.max(sigmoid, axis=1)
    output[rows] = I.cast(indices, I.i64)


def build(context):
    artifact = context.compile("sigmoid_argmax", _sigmoid_argmax_kernel, constexprs={})

    def sigmoid_argmax(input, dim=None, keepdim=False):
        return artifact.run(input)

    return sigmoid_argmax
