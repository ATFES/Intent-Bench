from __future__ import annotations

import intent
import torch
import intent.language as I


@intent.kernel
def _fused_index_select_eq_scalar(
    input: I.In[I.f32, ("INPUT_ROWS", "COLS")],
    index: I.In[I.i64, ("SELECTED_ROWS",)],
    other: I.f32,
    output: I.Out[I.bool, ("SELECTED_ROWS", "COLS")],
) -> None:
    selected_rows = I.domain(0, output.shape[0])
    columns = I.domain(0, output.shape[1])

    source_rows = index[I.indices(selected_rows)]
    source_columns = I.reshape(
        I.indices(columns),
        (1, output.shape[1]),
    )
    selected = input[source_rows, source_columns]
    output[selected_rows, columns] = selected == other


def build(context):
    kernel = context.compile("fused_index_select_eq_scalar", _fused_index_select_eq_scalar)

    def fused_index_select_eq(input, dim, index, other, *, out=None):
        if dim != 0:
            raise ValueError("this specialization supports dim=0")
        if not isinstance(other, (float, int)):
            raise ValueError("this specialization requires a scalar other")

        if out is None:
            output_shape = list(input.shape)
            output_shape[dim] = index.shape[0]
            output = torch.empty(
                output_shape,
                dtype=torch.bool,
                device=input.device,
            )
            return kernel(input, index, other, output)

        kernel(input, index, other, out)
        return out

    return fused_index_select_eq
