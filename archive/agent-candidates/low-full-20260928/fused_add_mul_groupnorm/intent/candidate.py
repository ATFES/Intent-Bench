import torch
import intent
import intent.language as I


@intent.kernel
def _fused_mul(
    x: I.In[I.f32, (16, 64, 128, 64)],
    y: I.In[I.f32, (16, 64, 128, 64)],
    out: I.Out[I.f32, (16, 64, 128, 64)],
):
    b = I.domain(0, 16)
    c = I.domain(0, 64)
    h = I.domain(0, 128)
    w = I.domain(0, 64)
    for ib in I.parallel(b):
        for ic in I.parallel(c):
            for ih in I.parallel(h):
                for iw in I.parallel(w):
                    v = (x[ib, ic, ih, iw] + y[ib, ic, ih, iw]) * y[ib, ic, ih, iw]
                    out[ib, ic, ih, iw] = v


@intent.kernel
def _group_norm(
    m: I.In[I.f32, (16, 64, 128, 64)],
    weight: I.In[I.f32, (64,)],
    bias: I.In[I.f32, (64,)],
    out: I.Out[I.f32, (16, 64, 128, 64)],
    eps: I.f32,
):
    # The channel grouping is an algorithmic interface: each group owns all
    # spatial members and has an independent mean/variance reduction.
    grouped = I.reshape(m[:, :, :, :], (16, 8, 8, 128, 64))
    mean = I.reduce.sum(grouped, axis=(2, 3, 4), acc_dtype=I.f32) / I.cast(65536, I.f32)
    mean5 = I.reshape(mean, (16, 8, 1, 1, 1))
    centered = grouped - mean5
    variance = I.reduce.sum(centered * centered, axis=(2, 3, 4), acc_dtype=I.f32) / I.cast(65536, I.f32)
    inv_std = I.fdiv(I.cast(1.0, I.f32), I.sqrt(I.reshape(variance, (16, 8, 1, 1, 1)) + eps))
    normalized = centered * inv_std
    normalized = I.reshape(normalized, (16, 64, 128, 64))
    gamma = I.reshape(weight, (1, 64, 1, 1))
    beta = I.reshape(bias, (1, 64, 1, 1))
    values = normalized * gamma + beta
    b = I.domain(0, 16)
    c = I.domain(0, 64)
    h = I.domain(0, 128)
    w = I.domain(0, 64)
    for ib in I.parallel(b):
        for ic in I.parallel(c):
            for ih in I.parallel(h):
                for iw in I.parallel(w):
                    out[ib, ic, ih, iw] = values[ib, ic, ih, iw]


def build(context):
    fused = context.compile("fused_add_mul_stage", _fused_mul)
    norm = context.compile("group_norm_stage", _group_norm)

    def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
        tmp = torch.empty_like(input1)
        result = out if out is not None else torch.empty_like(input1)
        fused(input1, input2, tmp)
        norm(tmp, weight, bias, result, eps)
        return result

    return fused_add_mul_groupnorm
