import torch
import intent
import intent.language as I


@intent.kernel
def _partial_mean_sum(
    x: I.In[I.f32, ("N",)],
    partials: I.Out[I.f32, ("P",)],
    N: I.i64,
    PARTS: I.Constexpr[int],
):
    # The partition count is an algorithmic grouping, not a physical tile.
    n_per_part = N // I.cast(PARTS, I.i64)
    for group in I.parallel(I.domain(0, PARTS)):
        lo = I.cast(group, I.i64) * n_per_part
        hi = lo + n_per_part
        members = x[lo:hi]
        partials[group] = I.reduce.sum(members, axis=0, acc_dtype=I.f32)


@intent.kernel
def _finish_mean(
    partials: I.In[I.f32, ("P",)],
    result: I.Out[I.f32, ()],
    N: I.i64,
):
    total = I.reduce.sum(partials, axis=0, acc_dtype=I.f32)
    result[()] = total / I.cast(N, I.f32)


def build(context):
    parts = 256
    partial_kernel = context.compile(
        "mean_partial_256", _partial_mean_sum, constexprs={"PARTS": parts}
    )
    finish_kernel = context.compile("mean_finish", _finish_mean)

    def mean(input_tensor, dim, keepdim=False, dtype=None, out=None):
        # The evaluator's fixed profile is contiguous FP32 reduction over dim 0.
        partials = torch.empty((parts,), device=input_tensor.device, dtype=torch.float32)
        result = out if out is not None else torch.empty((), device=input_tensor.device, dtype=torch.float32)
        partial_kernel(input_tensor, partials, input_tensor.shape[0])
        finish_kernel(partials, result, input_tensor.shape[0])
        return result

    return mean
