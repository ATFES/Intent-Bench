from __future__ import annotations

import intent
import intent.language as I


@intent.kernel
def gather_masked_fill_dim0(
    source: I.In[I.f32, ("SOURCE_ROWS", "COLUMNS")],
    indices: I.In[I.i64, ("ROWS", "COLUMNS")],
    mask: I.In[I.bool, ("ROWS", "COLUMNS")],
    output: I.Out[I.f32, ("ROWS", "COLUMNS")],
    fill_value: I.f32,
) -> None:
    for row in I.parallel(I.domain(0, indices.shape[0])):
        for column in I.parallel(I.domain(0, indices.shape[1])):
            source_row = indices[row, column]
            gathered = source[source_row, column]
            output[row, column] = I.select(
                mask[row, column], fill_value, gathered
            )


@intent.kernel
def gather_masked_fill_dim1(
    source: I.In[I.f32, ("ROWS", "SOURCE_COLUMNS")],
    indices: I.In[I.i64, ("ROWS", "COLUMNS")],
    mask: I.In[I.bool, ("ROWS", "COLUMNS")],
    output: I.Out[I.f32, ("ROWS", "COLUMNS")],
    fill_value: I.f32,
) -> None:
    for row in I.parallel(I.domain(0, indices.shape[0])):
        for column in I.parallel(I.domain(0, indices.shape[1])):
            source_column = indices[row, column]
            gathered = source[row, source_column]
            output[row, column] = I.select(
                mask[row, column], fill_value, gathered
            )


def build(context):
    dim0_kernel = context.compile(
        "fused_gather_masked_fill_dim0", gather_masked_fill_dim0
    )
    dim1_kernel = context.compile(
        "fused_gather_masked_fill_dim1", gather_masked_fill_dim1
    )

    def fused_gather_masked_fill(
        input,
        dim,
        index,
        mask,
        value,
        *,
        sparse_grad=False,
        out=None,
    ):
        if dim < 0:
            dimension = dim + input.ndim
        else:
            dimension = dim

        if dimension == 0:
            kernel = dim0_kernel
        elif dimension == 1:
            kernel = dim1_kernel
        else:
            raise ValueError("gather dimension is outside the two-dimensional profile")

        if out is None:
            return kernel.run(input, index, mask, value)

        kernel(input, index, mask, value, out)
        return out

    return fused_gather_masked_fill
