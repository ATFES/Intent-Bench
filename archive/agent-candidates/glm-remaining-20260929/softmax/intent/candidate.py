from __future__ import annotations

import intent
import intent.language as I
import torch


@intent.kernel
def softmax_rows(
    input: I.In[I.f32, ("M", "N"), I.constraints(noalias=True)],
    output: I.Out[I.f32, ("M", "N"), I.constraints(noalias=True)],
) -> None:
    for row in I.parallel(I.domain(0, input.shape[0])):
        row_values = input[row, :]
        row_maximum = I.reduce.max(row_values, axis=0)
        stabilized = row_values - row_maximum
        exponentials = I.exp2(stabilized * I.cast(1.4426950408889634, I.f32), approximate=True)
        denominator = I.reduce.sum(exponentials, axis=0, acc_dtype=I.f32)
        output[row, :] = I.fdiv(exponentials, denominator, approximate=True)


def build(context):
    softmax_rows_artifact = context.compile("softmax_rows_f32_dim1", softmax_rows)

    def softmax(
        input: "torch.Tensor",
        dim: "int",
        dtype: "torch.dtype | None" = None,
    ) -> "torch.Tensor":
        if dtype is not None and dtype != torch.float32:
            raise ValueError("this specialization supports float32 softmax only")
        return softmax_rows_artifact.run(input)

    return softmax
