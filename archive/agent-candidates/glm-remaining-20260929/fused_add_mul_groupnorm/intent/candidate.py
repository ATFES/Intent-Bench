from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def _fused_stats_kernel(
    input1: I.In[I.f32, (16, 8, 8, 8192)],
    input2: I.In[I.f32, (16, 8, 8, 8192)],
    mean: I.Out[I.f32, (16, 8)],
    rstd: I.Out[I.f32, (16, 8)],
    eps: I.f32,
) -> None:
    batch = I.domain(0, 16)
    groups = I.domain(0, 8)
    group_channels = I.domain(0, 8)
    spatial = I.domain(0, 8192)

    multiplied = (input1[batch, groups, group_channels, spatial] + input2[batch, groups, group_channels, spatial]) * input2[batch, groups, group_channels, spatial]
    group_mean = I.reduce.sum(multiplied, axis=(2, 3), acc_dtype=I.f32) / I.cast(65536, I.f32)
    group_mean_square = I.reduce.sum(multiplied * multiplied, axis=(2, 3), acc_dtype=I.f32) / I.cast(65536, I.f32)
    group_variance = group_mean_square - group_mean * group_mean

    mean[batch, groups] = group_mean
    rstd[batch, groups] = I.cast(1.0, I.f32) / I.sqrt(group_variance + eps)


@intent.kernel
def _fused_groupnorm_kernel(
    input1: I.In[I.f32, (16, 8, 8, 8192)],
    input2: I.In[I.f32, (16, 8, 8, 8192)],
    mean: I.In[I.f32, (16, 8)],
    rstd: I.In[I.f32, (16, 8)],
    weight: I.In[I.f32, (8, 8)],
    bias: I.In[I.f32, (8, 8)],
    output: I.Out[I.f32, (16, 8, 8, 8192)],
) -> None:
    batch = I.domain(0, 16)
    groups = I.domain(0, 8)
    group_channels = I.domain(0, 8)
    spatial = I.domain(0, 8192)

    multiplied = (input1[batch, groups, group_channels, spatial] + input2[batch, groups, group_channels, spatial]) * input2[batch, groups, group_channels, spatial]
    normalized = (multiplied - I.reshape(mean[batch, groups], (16, 8, 1, 1))) * I.reshape(rstd[batch, groups], (16, 8, 1, 1))
    affine = normalized * I.reshape(weight[groups, group_channels], (1, 8, 8, 1)) + I.reshape(bias[groups, group_channels], (1, 8, 8, 1))
    output[batch, groups, group_channels, spatial] = affine


def build(context):
    stats = context.compile(
        "fused_add_mul_groupnorm_stats",
        _fused_stats_kernel,
    )
    normalize = context.compile(
        "fused_add_mul_groupnorm_normalize",
        _fused_groupnorm_kernel,
    )

    def fused_add_mul_groupnorm(
        input1,
        input2,
        weight,
        bias,
        num_groups,
        eps=1e-05,
        *,
        out=None,
    ):
        batch, channels, depth, width = input1.shape
        group_channels = channels // num_groups
        spatial = depth * width
        input1_grouped = input1.view(batch, num_groups, group_channels, spatial)
        input2_grouped = input2.view(batch, num_groups, group_channels, spatial)
        mean = input1.new_empty((batch, num_groups))
        rstd = input1.new_empty((batch, num_groups))
        result = input1.new_empty(input1.shape) if out is None else out
        result_grouped = result.view(batch, num_groups, group_channels, spatial)
        weight_grouped = weight.view(num_groups, group_channels)
        bias_grouped = bias.view(num_groups, group_channels)

        stats(input1_grouped, input2_grouped, mean, rstd, eps)
        normalize(
            input1_grouped,
            input2_grouped,
            mean,
            rstd,
            weight_grouped,
            bias_grouped,
            result_grouped,
        )
        return result

    return fused_add_mul_groupnorm
