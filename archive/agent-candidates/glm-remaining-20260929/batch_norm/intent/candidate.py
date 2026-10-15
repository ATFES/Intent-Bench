from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def batch_norm_scale(
    running_mean: I.In[I.f32, ("CHANNEL",)],
    running_var: I.In[I.f32, ("CHANNEL",)],
    scale: I.Out[I.f32, ("CHANNEL",)],
    eps: I.f32,
) -> None:
    for channel in I.parallel(I.domain(0, running_var.shape[0])):
        variance = running_var[channel]
        scale[channel] = I.fdiv(
            I.cast(1.0, I.f32), I.sqrt(variance + eps)
        )


@intent.kernel
def batch_norm_inference(
    input: I.In[I.f16, ("BATCH", "CHANNEL", "HEIGHT", "WIDTH")],
    scale: I.In[I.f32, ("CHANNEL",)],
    output: I.Out[I.f16, ("BATCH", "CHANNEL", "HEIGHT", "WIDTH")],
    weight: I.In[I.f32, ("CHANNEL",)],
    bias: I.In[I.f32, ("CHANNEL",)],
    USE_WEIGHT: I.Constexpr[bool],
    USE_BIAS: I.Constexpr[bool],
) -> None:
    for batch in I.parallel(I.domain(0, input.shape[0])):
        for channel in I.parallel(I.domain(0, input.shape[1])):
            for height in I.parallel(I.domain(0, input.shape[2])):
                for width in I.parallel(I.domain(0, input.shape[3])):
                    value = I.cast(input[batch, channel, height, width], I.f32)
                    mean = running_mean[channel]
                    normalized = (value - mean) * scale[channel]

                    if USE_WEIGHT:
                        normalized = normalized * weight[channel]
                    if USE_BIAS:
                        normalized = normalized + bias[channel]

                    output[batch, channel, height, width] = I.cast(
                        normalized, I.f16
                    )


def build(context):
    variants = {}
    for use_weight in (False, True):
        for use_bias in (False, True):
            name = f"batch_norm_inference_{int(use_weight)}{int(use_bias)}"
            variants[(use_weight, use_bias)] = context.compile(
                name,
                batch_norm_inference,
                constexprs={
                    "USE_WEIGHT": use_weight,
                    "USE_BIAS": use_bias,
                },
            )
    scale_kernel = context.compile("batch_norm_scale", batch_norm_scale)

    def batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-05,
    ):
        output = input.new_empty(input.shape)
        invstd = torch.empty(
            running_var.shape, dtype=torch.float32, device=running_var.device
        )
        use_weight = weight is not None
        use_bias = bias is not None
        weight_arg = weight if use_weight else running_var
        bias_arg = bias if use_bias else running_var
        scale_kernel(running_var, invstd, eps)
        variants[(use_weight, use_bias)](
            input,
            invstd,
            output,
            running_mean,
            weight_arg,
            bias_arg,
        )
        return output

    return batch_norm
