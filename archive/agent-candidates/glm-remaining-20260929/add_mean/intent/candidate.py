from __future__ import annotations

import intent
import intent.language as I
import torch


GROUPS = 4096


@intent.kernel
def segment_sums_kernel(
    input: I.In[I.f32, ("N",)],
    other: I.In[I.f32, ("N",)],
    partial_sums: I.Out[I.f64, ("GROUPS",)],
    alpha: I.f32,
    N: I.Constexpr[int],
    GROUPS: I.Constexpr[int],
) -> None:
    segment_size = N // GROUPS
    input_segments = I.reshape(input, (GROUPS, segment_size))
    other_segments = I.reshape(other, (GROUPS, segment_size))
    combined = input_segments + alpha * other_segments
    segment_sums = I.reduce.sum(
        I.cast(combined, I.f64),
        axis=1,
        acc_dtype=I.f64,
    )
    group_domain = I.domain(0, GROUPS)
    partial_sums[group_domain] = segment_sums


@intent.kernel
def final_mean_kernel(
    partial_sums: I.In[I.f64, ("GROUPS",)],
    output: I.Out[I.f32, ()],
    N: I.Constexpr[int],
    GROUPS: I.Constexpr[int],
) -> None:
    total = I.reduce.sum(partial_sums, axis=0, acc_dtype=I.f64)
    mean = I.fdiv(total, I.cast(N, I.f64))
    output[()] = I.cast(mean, I.f32)


def build(context):
    segment_sums = context.compile(
        "add_mean_segment_sums",
        segment_sums_kernel,
        constexprs={"N": 1048576, "GROUPS": GROUPS},
    )
    final_mean = context.compile(
        "add_mean_final_mean",
        final_mean_kernel,
        constexprs={"N": 1048576, "GROUPS": GROUPS},
    )

    def add_mean(
        input,
        other,
        dim=None,
        alpha=1,
        keepdim=False,
        dtype=None,
        out=None,
    ):
        partial_tensor = input.new_empty((GROUPS,), dtype=torch.float64)
        output_tensor = out if out is not None else input.new_empty(())
        segment_sums(input, other, partial_tensor, float(alpha))
        final_mean(partial_tensor, output_tensor)
        return output_tensor

    return add_mean
