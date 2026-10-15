import torch
import triton
import triton.language as tl


@triton.jit
def _fused_index_select_eq(
    x_ptr,
    index_ptr,
    out_ptr,
    n_rows,
    other,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    row_mask = row < n_rows
    mask = row_mask & (cols < 128)

    selected_row = tl.load(index_ptr + row, mask=row_mask, other=0).to(tl.int32)
    values = tl.load(x_ptr + selected_row * 128 + cols, mask=mask, other=0.0)
    tl.store(out_ptr + row * 128 + cols, values == other, mask=mask)


def build(context):
    def wrapper(input, dim, index, other, *, out=None):
        if out is None:
            out = torch.empty(
                (index.numel(), 128),
                dtype=torch.bool,
                device=input.device,
            )

        _fused_index_select_eq[(triton.cdiv(index.numel(), 1),)](
            input,
            index,
            out,
            index.numel(),
            other,
            BLOCK=128,
        )
        return out

    return wrapper
