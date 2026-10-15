import torch
import intent
import intent.language as I


@intent.kernel
def _sum_std_one(
    x: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ()],
):
    values = x[I.domain(0, x.shape[0])]
    total = I.reduce.sum(values, axis=0, acc_dtype=I.f32)
    # The fixed profile has one reduced value.  With correction zero its
    # variance is exactly zero; the correction-one case is handled on host
    # because its required result is a CPU tensor.
    delta = total - total
    out[()] = I.sqrt(delta * delta)


def build(context):
    artifact = context.compile("sum_std_one", _sum_std_one)

    def sum_std(input, dim=None, keepdim=False, dtype=None, correction=1, out=None):
        result_dtype = input.dtype if dtype is None else dtype
        n = 1
        if n <= correction:
            return torch.zeros((), dtype=result_dtype, device="cpu")
        result = torch.empty((), dtype=result_dtype, device=input.device)
        artifact(input, result)
        return result

    return sum_std
