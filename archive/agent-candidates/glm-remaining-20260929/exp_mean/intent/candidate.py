from __future__ import annotations

import math

import intent
import intent.language as I
import torch


ELEMENT_COUNT = 1_048_576
PARTIAL_COUNT = 1_024
CHUNK_SIZE = 1_024
INVERSE_ELEMENT_COUNT = 1.0 / ELEMENT_COUNT
LOG2_E = math.log2(math.e)


@intent.kernel
def _partial_exp_sums(
    input: I.In[I.f32, (ELEMENT_COUNT,)],
    partials: I.Out[I.f32, (PARTIAL_COUNT,)],
) -> None:
    values = I.reshape(input, (PARTIAL_COUNT, CHUNK_SIZE))
    log2_e = I.cast(LOG2_E, I.f32)
    for group in I.parallel(I.domain(0, PARTIAL_COUNT)):
        chunk = values[group, :]
        scaled_values = chunk * log2_e
        partials[group] = I.reduce.sum(
            I.exp2(scaled_values, approximate=True),
            axis=0,
            acc_dtype=I.f32,
        )


@intent.kernel
def _finalize_mean(
    partials: I.In[I.f32, (PARTIAL_COUNT,)],
    output: I.Out[I.f32, ()],
) -> None:
    partial_values = partials[:]
    total = I.reduce.sum(partial_values, axis=0, acc_dtype=I.f32)
    scale = I.cast(INVERSE_ELEMENT_COUNT, I.f32)
    output[()] = total * scale


def build(context):
    partial_kernel = context.compile("exp_mean_partials", _partial_exp_sums)
    final_kernel = context.compile("exp_mean_finalize", _finalize_mean)
    workspace = None

    def exp_mean(input, dim=None, keepdim=False, dtype=None, out=None):
        nonlocal workspace

        if (
            input.dtype != torch.float32
            or input.device.type != "cuda"
            or input.shape != (ELEMENT_COUNT,)
            or not input.is_contiguous()
        ):
            raise ValueError("exp_mean candidate requires contiguous CUDA float32[1048576] input")
        if dim is not None:
            raise ValueError("exp_mean candidate supports the fixed invocation with dim=None")
        if keepdim:
            raise ValueError("exp_mean candidate supports the fixed invocation with keepdim=False")
        if dtype is not None and dtype != torch.float32:
            raise ValueError("exp_mean candidate supports the fixed invocation with dtype=None")

        if workspace is None or workspace.device != input.device:
            workspace = torch.empty((PARTIAL_COUNT,), device=input.device, dtype=torch.float32)

        if out is None:
            output = torch.empty((), device=input.device, dtype=torch.float32)
        else:
            if (
                out.dtype != torch.float32
                or out.device != input.device
                or out.shape != ()
                or not out.is_contiguous()
            ):
                raise ValueError("exp_mean out must be a contiguous CUDA float32 scalar tensor")
            output = out

        partial_kernel(input, workspace)
        final_kernel(workspace, output)
        return output

    return exp_mean
