import torch
import intent
import intent.language as I


@intent.kernel
def partial_moments(
    input: I.In[I.f32, (1048576,)],
    means: I.Out[I.f32, (4096,)],
    moments: I.Out[I.f32, (4096,)],
):
    values = I.reshape(input[:], (4096, 256))
    local_mean = I.reduce.sum(values, axis=1, acc_dtype=I.f32) / 256.0
    centered = values - I.reshape(local_mean, (4096, 1))
    means[:] = local_mean
    moments[:] = I.reduce.sum(centered * centered, axis=1, acc_dtype=I.f32)


@intent.kernel
def combine_moments(
    means: I.In[I.f32, (4096,)],
    moments: I.In[I.f32, (4096,)],
    output: I.Out[I.f32, ()],
    correction: I.f32,
):
    local_means = means[:]
    mean = I.reduce.sum(local_means, axis=0, acc_dtype=I.f32) / 4096.0
    delta = local_means - mean
    centered_moments = moments[:] + 256.0 * delta * delta
    total = I.reduce.sum(centered_moments, axis=0, acc_dtype=I.f32)
    denominator = I.maximum(1048576.0 - correction, 0.0)
    output[()] = I.sqrt(total / denominator)


def build(context):
    partial = context.compile("partial_moments", partial_moments)
    combine = context.compile("combine_moments", combine_moments)

    def wrapper(input, dim=None, *, correction=1, keepdim=False, out=None):
        means = torch.empty((4096,), dtype=input.dtype, device=input.device)
        moments = torch.empty_like(means)
        if out is None:
            out = torch.empty((1,) if keepdim else (), dtype=input.dtype,
                              device=input.device)
        partial(input, means, moments)
        combine(means, moments, out.reshape(()), float(correction))
        return out

    return wrapper
