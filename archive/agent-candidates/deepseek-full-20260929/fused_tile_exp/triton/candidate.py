import torch
import triton
import triton.language as tl


@triton.jit
def _fused_tile_exp_1d_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.exp(x)

    for r in tl.static_range(ROWS):
        tl.store(out_ptr + r * n_elements + offsets, y, mask=mask)


def build(context):
    def fused_tile_exp(input, dims, *, out=None):
        dims = tuple(int(d) for d in dims)
        rows = dims[0]

        if out is None:
            out = torch.empty(
                (rows,) + tuple(input.shape),
                dtype=input.dtype,
                device=input.device,
            )

        n_elements = input.numel()
        block = 1024
        grid = (triton.cdiv(n_elements, block),)
        _fused_tile_exp_1d_kernel[grid](
            input,
            out,
            n_elements,
            ROWS=rows,
            BLOCK=block,
        )
        return out

    return fused_tile_exp
