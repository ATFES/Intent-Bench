import torch
import intent
import intent.language as I


@intent.kernel
def _softmax_kernel(
    x: I.In[I.f32, ("M", "N")],
    out: I.Out[I.f32, ("M", "N")],
):
    M, N = x.shape
    values = x[:, :]
    row_max = I.reduce.max(values, axis=1, acc_dtype=I.f32)
    centered = values - I.reshape(row_max, (M, 1))
    numerators = I.exp(centered)
    row_sum = I.reduce.sum(numerators, axis=1, acc_dtype=I.f32)
    probabilities = numerators / I.reshape(row_sum, (M, 1))
    out[:, :] = probabilities


def build(context):
    softmax_artifact = context.compile("softmax_rowwise", _softmax_kernel)

    def softmax(input, dim, dtype=None):
        result = torch.empty_like(input)
        softmax_artifact(input, result)
        return result

    return softmax
