import torch
import triton
import triton.language as tl


@triton.jit
def _fused_add_mul_groupnorm(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    CHANNELS: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    GROUP_ELEMS: tl.constexpr,
    GROUP_CHANNELS: tl.constexpr,
    SPATIAL: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    group_id = tl.program_id(0)
    sample = group_id // NUM_GROUPS
    group = group_id % NUM_GROUPS
    base = sample * CHANNELS * SPATIAL + group * GROUP_CHANNELS * SPATIAL

    total = 0.0
    total_sq = 0.0
    for start in range(0, GROUP_ELEMS, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < GROUP_ELEMS
        x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + base + offsets, mask=mask, other=0.0)
        value = (x + y) * y
        value = value.to(tl.float32)
        total += tl.sum(tl.where(mask, value, 0.0), axis=0)
        total_sq += tl.sum(tl.where(mask, value * value, 0.0), axis=0)

    count = GROUP_ELEMS
    mean = total / count
    variance = total_sq / count - mean * mean
    inv_std = tl.rsqrt(variance + EPS)

    for start in range(0, GROUP_ELEMS, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < GROUP_ELEMS
        x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + base + offsets, mask=mask, other=0.0)
        value = ((x + y) * y).to(tl.float32)
        channel = group * GROUP_CHANNELS + offsets // SPATIAL
        gamma = tl.load(weight_ptr + channel, mask=mask, other=0.0)
        beta = tl.load(bias_ptr + channel, mask=mask, other=0.0)
        result = (value - mean) * inv_std * gamma + beta
        tl.store(out_ptr + base + offsets, result, mask=mask)


def build(context):
    def wrapper(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
        output = out if out is not None else torch.empty_like(input1)
        spatial = input1.shape[2] * input1.shape[3]
        channels = input1.shape[1]
        group_channels = channels // num_groups
        group_elems = group_channels * spatial
        grid = (input1.shape[0] * num_groups,)
        _fused_add_mul_groupnorm[grid](
            input1,
            input2,
            weight,
            bias,
            output,
            CHANNELS=channels,
            NUM_GROUPS=num_groups,
            GROUP_ELEMS=group_elems,
            GROUP_CHANNELS=group_channels,
            SPATIAL=spatial,
            EPS=eps,
            BLOCK=1024,
            num_warps=8,
        )
        return output

    return wrapper
