import torch
import intent
import intent.language as I


@intent.kernel
def _exp_stage(
    x: I.In[I.f32, ("N",)],
    y: I.Out[I.f32, ("N",)],
):
    n = x.shape[0]
    axis = I.domain(0, n)
    y[axis] = I.exp(x[axis])


@intent.kernel
def _mean_stage(
    values: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ()],
):
    total = I.reduce.sum(values[I.domain(0, values.shape[0])], axis=0, acc_dtype=I.f32)
    out[()] = total / I.cast(values.shape[0], I.f32)


def build(context):
    exp_artifact = context.compile("exp_mean_exp", _exp_stage)
    mean_artifact = context.compile("exp_mean_reduce", _mean_stage)

    def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
        values = torch.empty_like(input)
        exp_artifact(input, values)
        result = out if out is not None else torch.empty((), device=input.device, dtype=input.dtype)
        mean_artifact(values, result)
        return result

    return exp_mean
