from __future__ import annotations

import intent
import intent.language as I
import torch


GROUPS = 512
SEGMENT_SIZE = 2048


@intent.kernel
def segment_moments(
    values: I.In[I.f32, ("N",)],
    means: I.Out[I.f64, ("GROUPS",)],
    squared_deviations: I.Out[I.f64, ("GROUPS",)],
    GROUPS: I.Constexpr[int],
    SEGMENT_SIZE: I.Constexpr[int],
):
    for segment in I.parallel(I.domain(0, GROUPS)):
        members = I.domain(segment * SEGMENT_SIZE, (segment + 1) * SEGMENT_SIZE)
        segment_values = I.cast(values[members], I.f64)
        segment_count = I.cast(SEGMENT_SIZE, I.f64)
        segment_mean = I.reduce.sum(segment_values, axis=0, acc_dtype=I.f64) / segment_count
        centered = segment_values - segment_mean
        segment_squared_deviation = I.reduce.sum(
            centered * centered, axis=0, acc_dtype=I.f64
        )
        means[segment] = segment_mean
        squared_deviations[segment] = segment_squared_deviation


@intent.fn
def merge_moments(left, right):
    left_count, left_mean, left_m2 = left
    right_count, right_mean, right_m2 = right
    count = left_count + right_count
    mean_shift = (right_mean - left_mean) * (right_count / count)
    mean = left_mean + mean_shift
    m2 = (
        left_m2
        + right_m2
        + mean_shift
        * mean_shift
        * left_count
        * right_count
        / count
    )
    return count, mean, m2


@intent.kernel
def merge_and_normalize(
    means: I.In[I.f64, ("GROUPS",)],
    squared_deviations: I.In[I.f64, ("GROUPS",)],
    result: I.Out[I.f32, ()],
    GROUPS: I.Constexpr[int],
    SEGMENT_SIZE: I.Constexpr[int],
    CORRECTION: I.Constexpr[int],
):
    counts = I.full((GROUPS,), SEGMENT_SIZE, dtype=I.f64)
    identity = (
        I.cast(0, I.f64),
        I.cast(0, I.f64),
        I.cast(0, I.f64),
    )
    total_count, total_mean, total_m2 = I.reduce(
        (counts, means, squared_deviations),
        axis=0,
        identity=identity,
        combine=merge_moments,
    )
    denominator = I.maximum(
        total_count - I.cast(CORRECTION, I.f64), I.cast(0, I.f64)
    )
    variance = I.maximum(total_m2 / denominator, I.cast(0, I.f64))
    result[()] = I.cast(I.sqrt(variance), I.f32)


def build(context):
    segment_artifact = context.compile(
        "std_segment_moments",
        segment_moments,
        constexprs={
            "N": GROUPS * SEGMENT_SIZE,
            "GROUPS": GROUPS,
            "SEGMENT_SIZE": SEGMENT_SIZE,
        },
    )
    merge_artifact = context.compile(
        "std_merge_and_normalize",
        merge_and_normalize,
        constexprs={
            "GROUPS": GROUPS,
            "SEGMENT_SIZE": SEGMENT_SIZE,
            "CORRECTION": 1,
        },
    )

    def std(input, dim=None, correction=1, keepdim=False, out=None):
        result = out
        if result is None:
            result = torch.empty((), device=input.device, dtype=torch.float32)
        means = torch.empty((GROUPS,), device=input.device, dtype=torch.float64)
        squared_deviations = torch.empty(
            (GROUPS,), device=input.device, dtype=torch.float64
        )
        segment_artifact(input, means, squared_deviations)
        merge_artifact(means, squared_deviations, result)
        return result

    return std

