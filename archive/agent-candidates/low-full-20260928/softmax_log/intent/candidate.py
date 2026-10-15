import torch
import intent
import intent.language as I


@intent.kernel
def _log_stage(
    x: I.In[I.f32, ("M", "N")],
    y: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, x.shape[0])
    cols = I.domain(0, x.shape[1])
    y[rows, cols] = I.log(x[rows, cols])


@intent.kernel
def _softmax_stage(
    x: I.In[I.f32, ("M", "N")],
    y: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, x.shape[0])
    cols = I.domain(0, x.shape[1])
    values = x[rows, cols]
    row_max = I.reduce.max(values, axis=1, acc_dtype=I.f32)
    row_max = I.reshape(row_max, (x.shape[0], 1))
    weights = I.exp(values - row_max)
    row_sum = I.reduce.sum(weights, axis=1, acc_dtype=I.f32)
    row_sum = I.reshape(row_sum, (x.shape[0], 1))
    y[rows, cols] = weights / row_sum


def build(context):
    log_artifact = context.compile("softmax_log_log", _log_stage)
    softmax_artifact = context.compile("softmax_log_softmax", _softmax_stage)

    def softmax_log(input, dim=-1, dtype=None):
        if dtype is not None:
            raise TypeError("dtype variants are not available for the fixed profile")
        if dim not in (-1, 1):
            raise ValueError("fixed profile supports dim=-1")
        logged = torch.empty_like(input)
        output = torch.empty_like(input)
        log_artifact(input, logged)
        softmax_artifact(logged, output)
        return output

    return softmax_log
