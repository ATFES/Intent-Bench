from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def _sigmoid_batch_norm_inference(
    input: I.In[I.f32, ("ROWS", "CHANNELS")],
    running_mean: I.In[I.f32, ("CHANNELS",)],
    running_var: I.In[I.f32, ("CHANNELS",)],
    weight: I.In[I.f32, ("CHANNELS",)],
    bias: I.In[I.f32, ("CHANNELS",)],
    output: I.Out[I.f32, ("ROWS", "CHANNELS")],
    eps: I.f32,
    HAS_WEIGHT: I.Constexpr[bool],
    HAS_BIAS: I.Constexpr[bool],
) -> None:
    values = input[:, :]
    mean = running_mean[:]
    scale = I.rsqrt(running_var[:] + eps)
    normalized = (values - mean) * scale

    if HAS_WEIGHT:
        normalized = normalized * weight[:]
    if HAS_BIAS:
        normalized = normalized + bias[:]

    output[:, :] = I.sigmoid(normalized)


@intent.kernel
def _sigmoid_batch_norm_training(
    input: I.In[I.f32, ("ROWS", "CHANNELS")],
    running_mean: I.InOut[I.f32, ("CHANNELS",)],
    running_var: I.InOut[I.f32, ("CHANNELS",)],
    weight: I.In[I.f32, ("CHANNELS",)],
    bias: I.In[I.f32, ("CHANNELS",)],
    output: I.Out[I.f32, ("ROWS", "CHANNELS")],
    count: I.f32,
    momentum: I.f32,
    eps: I.f32,
    HAS_WEIGHT: I.Constexpr[bool],
    HAS_BIAS: I.Constexpr[bool],
) -> None:
    values = input[:, :]
    one = I.cast(1.0, I.f32)
    batch_mean = I.reduce.sum(values, axis=0) / count
    squared_error = (values - batch_mean) * (values - batch_mean)
    batch_variance = I.reduce.sum(squared_error, axis=0) / count
    unbiased_scale = count / (count - one)
    unbiased_variance = batch_variance * unbiased_scale

    running_mean[:] = (
        (one - momentum) * running_mean[:] + momentum * batch_mean
    )
    running_var[:] = (
        (one - momentum) * running_var[:] + momentum * unbiased_variance
    )

    scale = I.rsqrt(batch_variance + eps)
    normalized = (values - batch_mean) * scale
    if HAS_WEIGHT:
        normalized = normalized * weight[:]
    if HAS_BIAS:
        normalized = normalized + bias[:]

    output[:, :] = I.sigmoid(normalized)


def build(context):
    inference_false_false = context.compile(
        "sigmoid_batch_norm_inference_false_false",
        _sigmoid_batch_norm_inference,
        constexprs={"HAS_WEIGHT": False, "HAS_BIAS": False},
    )
    inference_false_true = context.compile(
        "sigmoid_batch_norm_inference_false_true",
        _sigmoid_batch_norm_inference,
        constexprs={"HAS_WEIGHT": False, "HAS_BIAS": True},
    )
    inference_true_false = context.compile(
        "sigmoid_batch_norm_inference_true_false",
        _sigmoid_batch_norm_inference,
        constexprs={"HAS_WEIGHT": True, "HAS_BIAS": False},
    )
    inference_true_true = context.compile(
        "sigmoid_batch_norm_inference_true_true",
        _sigmoid_batch_norm_inference,
        constexprs={"HAS_WEIGHT": True, "HAS_BIAS": True},
    )
    training_false_false = context.compile(
        "sigmoid_batch_norm_training_false_false",
        _sigmoid_batch_norm_training,
        constexprs={"HAS_WEIGHT": False, "HAS_BIAS": False},
    )
    training_false_true = context.compile(
        "sigmoid_batch_norm_training_false_true",
        _sigmoid_batch_norm_training,
        constexprs={"HAS_WEIGHT": False, "HAS_BIAS": True},
    )
    training_true_false = context.compile(
        "sigmoid_batch_norm_training_true_false",
        _sigmoid_batch_norm_training,
        constexprs={"HAS_WEIGHT": True, "HAS_BIAS": False},
    )
    training_true_true = context.compile(
        "sigmoid_batch_norm_training_true_true",
        _sigmoid_batch_norm_training,
        constexprs={"HAS_WEIGHT": True, "HAS_BIAS": True},
    )
    inference_artifacts = {
        (False, False): inference_false_false,
        (False, True): inference_false_true,
        (True, False): inference_true_false,
        (True, True): inference_true_true,
    }
    training_artifacts = {
        (False, False): training_false_false,
        (False, True): training_false_true,
        (True, False): training_true_false,
        (True, True): training_true_true,
    }

    def sigmoid_batch_norm(
        input,
        running_mean,
        running_var,
        weight=None,
        bias=None,
        training=False,
        momentum=0.1,
        eps=1e-5,
    ):
        channels = input.shape[1]
        flat_input = input.view(-1, channels) if input.ndim != 2 else input
        placeholder = running_mean
        selected_weight = weight if weight is not None else placeholder
        selected_bias = bias if bias is not None else placeholder
        specialization = (weight is not None, bias is not None)
        is_training = bool(training)

        if is_training:
            flat_output = training_artifacts[specialization].run(
                flat_input,
                running_mean,
                running_var,
                selected_weight,
                selected_bias,
                float(flat_input.shape[0]),
                float(momentum),
                float(eps),
            )
        else:
            flat_output = inference_artifacts[specialization].run(
                flat_input,
                running_mean,
                running_var,
                selected_weight,
                selected_bias,
                float(eps),
            )

        return flat_output.view(input.shape)

    return sigmoid_batch_norm
