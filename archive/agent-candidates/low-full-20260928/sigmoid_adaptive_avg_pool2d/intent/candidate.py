import torch
import intent
import intent.language as I


@intent.kernel
def _pool_sigmoid(
    input: I.In[I.f32, (2, 2, 16, 16)],
    output: I.Out[I.f32, (2, 2, 1, 1)],
):
    # Each (N, C) point owns an independent adaptive-pooling reduction.
    ns = I.domain(0, 2)
    cs = I.domain(0, 2)
    hs = I.domain(0, 16)
    ws = I.domain(0, 16)
    for n in I.parallel(ns):
        for c in I.parallel(cs):
            values = input[n, c, hs, ws]
            total = I.reduce.sum(values, axis=(0, 1), acc_dtype=I.f32)
            pooled = total / I.cast(256.0, I.f32)
            result = I.cast(1.0, I.f32) / (
                I.cast(1.0, I.f32) + I.exp(-pooled)
            )
            output[n, c, 0, 0] = result


def build(context):
    compiled = context.compile("pool_sigmoid", _pool_sigmoid)

    def sigmoid_adaptive_avg_pool2d(input: "Tensor", output_size):
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        if tuple(output_size) != (1, 1):
            raise ValueError("fixed profile supports output_size=(1, 1)")
        output = torch.empty((2, 2, 1, 1), device=input.device, dtype=input.dtype)
        compiled(input, output)
        return output

    return sigmoid_adaptive_avg_pool2d
