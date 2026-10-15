from __future__ import annotations

import torch
import intent
import intent.language as I


GROUPS = 1024
CHUNK_SIZE = 1024
TOTAL_SIZE = GROUPS * CHUNK_SIZE


@intent.kernel
def chunk_sums(
    input_tensor: I.In[I.f32, ("GROUPS", "CHUNK_SIZE")],
    partial_sums: I.Out[I.f32, ("GROUPS",)],
) -> None:
    groups = I.domain(0, input_tensor.shape[0])
    members = I.domain(0, input_tensor.shape[1])
    values = input_tensor[groups, members]
    partial_sums[groups] = I.reduce.sum(values, axis=1, acc_dtype=I.f32)


@intent.kernel
def finalize_mean(
    partial_sums: I.In[I.f32, ("GROUPS",)],
    output: I.Out[I.f32, ()],
    element_count: I.f32,
) -> None:
    groups = I.domain(0, partial_sums.shape[0])
    total = I.reduce.sum(partial_sums[groups], axis=0, acc_dtype=I.f32)
    output[()] = total / element_count


def build(context):
    chunk_sums_kernel = context.compile(
        "mean_chunk_sums_f32",
        chunk_sums,
        constexprs={"GROUPS": GROUPS, "CHUNK_SIZE": CHUNK_SIZE},
    )
    finalize_mean_kernel = context.compile(
        "mean_finalize_f32",
        finalize_mean,
    )
    scratch = torch.empty((GROUPS,), device="cuda", dtype=torch.float32)

    def mean(
        input_tensor,
        dim,
        keepdim=False,
        dtype=None,
        out=None,
    ):
        if (
            input_tensor.shape != (TOTAL_SIZE,)
            or input_tensor.dtype is not torch.float32
            or input_tensor.device.type != "cuda"
            or input_tensor.stride() != (1,)
            or dim != 0
            or keepdim
            or dtype is not None
            or (out is not None and (out.shape != () or out.dtype is not torch.float32))
        ):
            raise ValueError("candidate specializes the supplied mean invocation profile")

        if out is None:
            result = torch.empty((), device=input_tensor.device, dtype=torch.float32)
        else:
            result = out

        grouped_input = input_tensor.view(GROUPS, CHUNK_SIZE)
        chunk_sums_kernel(grouped_input, scratch)
        finalize_mean_kernel(scratch, result, float(TOTAL_SIZE))
        return result

    return mean
