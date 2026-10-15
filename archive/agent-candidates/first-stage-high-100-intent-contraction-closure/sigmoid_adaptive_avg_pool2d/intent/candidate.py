import intent
import intent.language as I


@intent.kernel
def _sigmoid_adaptive_avg_pool2d(
    input: I.In[I.f32, (2, 2, 16, 16)],
    output: I.Out[I.f32, (2, 2, 1, 1)],
):
    batch = I.domain(0, 2)
    channel = I.domain(0, 2)
    height = I.domain(0, 16)
    width = I.domain(0, 16)
    out_height = I.domain(0, 1)
    out_width = I.domain(0, 1)

    values = input[batch, channel, height, width]
    pooled = I.reduce.sum(values, axis=(2, 3), acc_dtype=I.f32)
    averaged = I.fdiv(pooled, I.cast(256.0, I.f32))
    activated = I.sigmoid(averaged)
    output[batch, channel, out_height, out_width] = I.reshape(
        activated, (2, 2, 1, 1)
    )


def build(context):
    artifact = context.compile(
        "sigmoid_adaptive_avg_pool2d", _sigmoid_adaptive_avg_pool2d, constexprs={}
    )

    def sigmoid_adaptive_avg_pool2d(input, output_size):
        return artifact.run(input)

    return sigmoid_adaptive_avg_pool2d
