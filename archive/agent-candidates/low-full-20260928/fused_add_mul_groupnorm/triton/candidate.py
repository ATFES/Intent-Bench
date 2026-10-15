import torch
import triton
import triton.language as tl


@triton.jit
def _fused_groupnorm(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    eps,
    n_elements_per_group,
    channels_per_group,
    height,
    width,
    groups,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    group_id = pid % groups
    batch_id = pid // groups

    spatial = height * width
    group_base = (batch_id * groups + group_id) * n_elements_per_group
    channel_base = group_id * channels_per_group

    mean = 0.0
    for start in range(0, 65536, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n_elements_per_group
        x = tl.load(x_ptr + group_base + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + group_base + offsets, mask=mask, other=0.0)
        m = (x + y) * y
        mean += tl.sum(tl.where(mask, m, 0.0), axis=0)
    mean = mean / n_elements_per_group

    variance = 0.0
    for start in range(0, 65536, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n_elements_per_group
        x = tl.load(x_ptr + group_base + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + group_base + offsets, mask=mask, other=0.0)
        m = (x + y) * y
        delta = m - mean
        variance += tl.sum(tl.where(mask, delta * delta, 0.0), axis=0)
    variance = variance / n_elements_per_group
    inv_std = tl.rsqrt(variance + eps)

    for start in range(0, 65536, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < n_elements_per_group
        x = tl.load(x_ptr + group_base + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + group_base + offsets, mask=mask, other=0.0)
        m = (x + y) * y
        channel = channel_base + offsets // spatial
        weight = tl.load(weight_ptr + channel, mask=mask, other=0.0)
        bias = tl.load(bias_ptr + channel, mask=mask, other=0.0)
        result = (m - mean) * inv_std * weight + bias
        tl.store(out_ptr + group_base + offsets, result, mask=mask)


def build(context):
    def fused_add_mul_groupnorm(input1, input2, weight, bias, num_groups, eps=1e-5, *, out=None):
        output = out if out is not None else torch.empty_like(input1)
        n, channels, height, width = input1.shape
        channels_per_group = channels // num_groups
        n_elements_per_group = channels_per_group * height * width
        _fused_groupnorm[(n * num_groups,)](
            input1,
            input2,
            weight,
            bias,
            output,
            eps,
            n_elements_per_group,
            channels_per_group,
            height,
            width,
            num_groups,
            BLOCK=1024,
        )
        return output

    return fused_add_mul_groupnorm
