import torch
import intent
import intent.language as I


@intent.kernel
def std_full_reduce(
    x: I.In[I.f32, (1048576,)],
    n: I.f32,
    out: I.Out[I.f32, ()],
    correction: I.Constexpr[int],
):
    # Two-pass algorithm for numerical stability:
    # Stage 1: compute mean = sum(x) / N
    total = I.reduce.sum(x, axis=0, acc_dtype=I.f32)
    mean = I.fdiv(total, n)

    # Stage 2: variance = sum((x - mean)^2) / max(0, N - correction)
    diff = x - mean
    ssq = I.reduce.sum(diff * diff, axis=0, acc_dtype=I.f32)
    raw_denom = n - I.cast(correction, I.f32)
    # Guard against N == correction (zero degrees of freedom)
    denom = I.maximum(raw_denom, I.cast(0.0, I.f32))
    var = I.fdiv(ssq, denom)

    out[()] = I.sqrt(var)


def build(context):
    kernel = context.compile(
        "std_full_reduce",
        std_full_reduce,
        constexprs={"correction": 1},
    )

    def std(input, dim=None, *, correction=1, keepdim=False, out=None):
        # Fast path for the fixed profile: full reduction of a 1-D float32 CUDA
        # tensor with default correction=1 and keepdim=False.
        if (
            dim is None
            and correction == 1
            and not keepdim
            and out is None
            and input.dtype == torch.float32
            and input.is_cuda
            and input.dim() == 1
            and input.shape[0] == 1048576
        ):
            return kernel.run(input, float(1048576))

        # General fallback for all other parameter combinations
        return torch.std(input, dim=dim, correction=correction,
                         keepdim=keepdim, out=out)

    return std

