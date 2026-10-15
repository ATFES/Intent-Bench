from __future__ import annotations

import torch
import intent
import intent.language as I


_GROUP_COUNT = 1024
_SEGMENT_SIZE = 1024


@intent.kernel
def partial_logsumexp(
    input: I.In[I.f32, (1024, 1024)],
    maxima: I.Out[I.f32, (1024,)],
    sums: I.Out[I.f32, (1024,)],
    GROUP_COUNT: I.Constexpr[int],
    SEGMENT_SIZE: I.Constexpr[int],
) -> None:
    groups = I.domain(0, GROUP_COUNT)
    segments = I.domain(0, SEGMENT_SIZE)
    values = input[groups, segments]
    group_maxima = I.reduce.max(values, axis=1)
    centered = values - I.reshape(group_maxima, (GROUP_COUNT, 1))
    group_sums = I.reduce.sum(I.exp(centered), axis=1)
    maxima[groups] = group_maxima
    sums[groups] = group_sums


@intent.kernel
def combine_logsumexp(
    maxima: I.In[I.f32, (1024,)],
    sums: I.In[I.f32, (1024,)],
    output: I.Out[I.f32, ()],
    GROUP_COUNT: I.Constexpr[int],
) -> None:
    groups = I.domain(0, GROUP_COUNT)
    values = maxima[groups]
    scales = sums[groups]
    global_max = I.reduce.max(values, axis=0)
    result = I.cast(0.0, I.f32)

    if global_max == I.cast(I.inf, I.f32):
        result = I.cast(I.inf, I.f32)
    elif global_max == I.cast(-I.inf, I.f32):
        result = I.cast(-I.inf, I.f32)
    else:
        global_sum = I.reduce.sum(
            scales * I.exp(values - global_max),
            axis=0,
        )
        result = global_max + I.log(global_sum)

    output[()] = result


def build(context):
    partial_artifact = context.compile(
        "logsumexp_partials",
        partial_logsumexp,
        constexprs={
            "GROUP_COUNT": _GROUP_COUNT,
            "SEGMENT_SIZE": _SEGMENT_SIZE,
        },
    )
    combine_artifact = context.compile(
        "logsumexp_combine",
        combine_logsumexp,
        constexprs={"GROUP_COUNT": _GROUP_COUNT},
    )

    def logsumexp(input, dim=0, keepdim=False, *, out=None):
        if input.ndim != 1 or dim != 0 or keepdim:
            raise ValueError("this specialization supports dim=0 with keepdim=False")
        if input.shape[0] != _GROUP_COUNT * _SEGMENT_SIZE:
            raise ValueError("input length does not match the compiled reduction profile")
        if out is None:
            out = torch.empty((), dtype=torch.float32, device=input.device)

        values = input.view(_GROUP_COUNT, _SEGMENT_SIZE)
        maxima = torch.empty(
            (_GROUP_COUNT,), dtype=torch.float32, device=input.device
        )
        sums = torch.empty((_GROUP_COUNT,), dtype=torch.float32, device=input.device)
        partial_artifact(values, maxima, sums)
        combine_artifact(maxima, sums, out)
        return out

    return logsumexp
