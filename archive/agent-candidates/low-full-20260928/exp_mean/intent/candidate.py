import torch
import intent
import intent.language as I


@intent.kernel
def _exp_mean_kernel(
    input: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ()],
):
    elements = I.domain(0, input.shape[0])
    values = I.exp(input[elements])
    total = I.reduce.sum(values, axis=0, acc_dtype=I.f32)
    count = I.cast(input.shape[0], I.f32)
    out[()] = total / count


def build(context):
    kernel = context.compile("exp_mean", _exp_mean_kernel)

    def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
        if out is None:
            result = torch.empty((), device=input.device, dtype=input.dtype)
        else:
            result = out
        kernel(input, result)
        return result

    return exp_mean
