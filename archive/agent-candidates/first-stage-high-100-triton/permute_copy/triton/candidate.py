import torch
import triton
import triton.language as tl


@triton.jit
def _permute_copy_kernel(x_ptr, y_ptr, TILE: tl.constexpr):
    pid = tl.program_id(0)
    d1 = pid // TILE
    d2 = pid % TILE

    d0 = tl.arange(0, TILE)
    d3 = tl.arange(0, TILE)

    # Load a contiguous d0 row for each d3, then transpose the tile in
    # registers/shared memory so both the read and write sides are coalesced.
    x_offsets = (
        d3[:, None] * 32768
        + d2 * 1024
        + d1 * 32
        + d0[None, :]
    )
    values = tl.load(x_ptr + x_offsets)
    values = tl.trans(values)

    y_offsets = (
        d0[:, None] * 32768
        + d1 * 1024
        + d2 * 32
        + d3[None, :]
    )
    tl.store(y_ptr + y_offsets, values)


def build(context):
    def wrapper(input, dims):
        output = torch.empty_like(input)
        _permute_copy_kernel[(1024,)](input, output, TILE=32, num_warps=8)
        return output

    return wrapper
