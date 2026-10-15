from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def _silu_batch_norm_eval(
    input: I.In[I.f32, ("N", "C", "H", "W")],
    running_mean: I.In[I.f32, ("C",)],
    running_var: I.In[I.f32, ("C",)],
    weight: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    has_weight: I.bool,
    has_bias: I.bool,
    eps: I.f32,
    output: I.Out[I.f32, ("N", "C", "H", "W")],
) -> None:
    channels = input.shape[1]
    values = input[:, :, :, :]
    means = running_mean[:]
    variances = running_var[:]
    rstds = I.rsqrt(variances + eps)
    normalized = (values - I.reshape(means, (1, channels, 1, 1))) * I.reshape(
        rstds, (1, channels, 1, 1)
    )

    if has_weight:
        scale = weight[:]
    else:
        scale = I.full((channels,), 1.0, I.f32)
    if has_bias:
        shift = bias[:]
    else:
        shift = I.full((channels,), 0.0, I.f32)

    activated_input = normalized * I.reshape(scale, (1, channels, 1, 1)) + I.reshape(
        shift, (1, channels, 1, 1)
    )
    output[:, :, :, :] = activated_input * I.sigmoid(activated_input)


@intent.kernel
def _batch_stats_and_running_update(
    input: I.In[I.f32, ("N", "C", "H", "W")],
    running_mean: I.InOut[I.f32, ("C",)],
    running_var: I.InOut[I.f32, ("C",)],
    count: I.f32,
    inv_count: I.f32,
    momentum: I.f32,
    eps: I.f32,
    batch_mean: I.Out[I.f32, ("C",)],
    batch_rstd: I.Out[I.f32, ("C",)],
) -> None:
    samples, channels, height, width = input.shape
    values = I.reshape(input, (samples, channels, height * width))
    moments = I.join(values, values * values)
    spatial_totals = I.reduce.sum(moments, axis=2)
    totals = I.reduce.sum(spatial_totals, axis=0)

    value_sum = totals[:, 0]
    squared_sum = totals[:, 1]
    means = value_sum * inv_count
    variances = squared_sum * inv_count - means * means
    rstds = I.rsqrt(variances + eps)

    old_running_mean = running_mean[:]
    old_running_var = running_var[:]
    running_mean[:] = old_running_mean + momentum * (means - old_running_mean)
    running_var[:] = old_running_var + momentum * (variances - old_running_var)
    batch_mean[:] = means
    batch_rstd[:] = rstds


@intent.kernel
def _silu_batch_norm_apply(
    input: I.In[I.f32, ("N", "C", "H", "W")],
    batch_mean: I.In[I.f32, ("C",)],
    batch_rstd: I.In[I.f32, ("C",)],
    weight: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    has_weight: I.bool,
    has_bias: I.bool,
    output: I.Out[I.f32, ("N", "C", "H", "W")],
) -> None:
    channels = input.shape[1]
    values = input[:, :, :, :]
    means = batch_mean[:]
    rstds = batch_rstd[:]
    normalized = (values - I.reshape(means, (1, channels, 1, 1))) * I.reshape(
        rstds, (1, channels, 1, 1)
    )

    if has_weight:
        scale = weight[:]
    else:
        scale = I.full((channels,), 1.0, I.f32)
    if has_bias:
        shift = bias[:]
    else:
        shift = I.full((channels,), 0.0, I.f32)

    activated_input = normalized * I.reshape(scale, (1, channels, 1, 1)) + I.reshape(
        shift, (1, channels, 1, 1)
    )
    output[:, :, :, :] = activated_input * I.sigmoid(activated_input)


def build(context):
    eval_artifact = context.compile(
        "silu_batch_norm_eval_f32", _silu_batch_norm_eval
    )
    stats_artifact = context.compile(
        "batch_stats_and_running_update_f32", _batch_stats_and_running_update
    )
    apply_artifact = context.compile(
        "silu_batch_norm_apply_f32", _silu_batch_norm_apply
    )

    def silu_batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-5,
    ):
        has_weight = weight is not None
        has_bias = bias is not None
        weight_view = weight if has_weight else running_mean
        bias_view = bias if has_bias else running_mean

        if training:
            sample_count = input.shape[0] * input.shape[2] * input.shape[3]
            inv_count = 1.0 / sample_count
            batch_mean, batch_rstd = stats_artifact.run(
                input,
                running_mean,
                running_var,
                float(sample_count),
                inv_count,
                momentum,
                eps,
            )
            return apply_artifact.run(
                input,
                batch_mean,
                batch_rstd,
                weight_view,
                bias_view,
                has_weight,
                has_bias,
            )

        return eval_artifact.run(
            input,
            running_mean,
            running_var,
            weight_view,
            bias_view,
            has_weight,
            has_bias,
            eps,
        )

    return silu_batch_norm
