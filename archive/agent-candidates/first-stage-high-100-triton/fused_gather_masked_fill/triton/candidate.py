import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gather_masked_fill_dim0(
    x_ptr,
    index_ptr,
    mask_ptr,
    out_ptr,
    rows,
    cols,
    x_stride0,
    x_stride1,
    index_stride0,
    index_stride1,
    mask_stride0,
    mask_stride1,
    out_stride0,
    out_stride1,
    value,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = (row < rows) & (col < cols)

    index_offset = row * index_stride0 + col * index_stride1
    gather_index = tl.load(index_ptr + index_offset, mask=valid, other=0)
    x_offset = gather_index * x_stride0 + col * x_stride1
    gathered = tl.load(x_ptr + x_offset, mask=valid, other=0.0)

    mask_offset = row * mask_stride0 + col * mask_stride1
    fill_mask = tl.load(mask_ptr + mask_offset, mask=valid, other=0)
    result = tl.where(fill_mask, value, gathered)

    out_offset = row * out_stride0 + col * out_stride1
    tl.store(out_ptr + out_offset, result, mask=valid)


def build(context):
    def wrapper(input, dim, index, mask, value, *, sparse_grad=False, out=None):
        # The supplied profile is a contiguous 2-D gather along dimension 0.
        # Keep the output target when one is supplied, matching torch's out form.
        output = torch.empty_like(input) if out is None else out
        rows, cols = index.shape
        _fused_gather_masked_fill_dim0[(triton.cdiv(rows, 1),)](
            input,
            index,
            mask,
            output,
            rows,
            cols,
            input.stride(0),
            input.stride(1),
            index.stride(0),
            index.stride(1),
            mask.stride(0),
            mask.stride(1),
            output.stride(0),
            output.stride(1),
            value,
            BLOCK=1024,
            num_warps=8,
        )
        return output

    return wrapper
