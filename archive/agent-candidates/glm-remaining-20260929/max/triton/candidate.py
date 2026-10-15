import torch
import triton
import triton.language as tl


@triton.jit
def _block_max_index_kernel(
    input_ptr,
    packed_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    values = tl.load(input_ptr + offsets)
    bits = values.to(tl.uint32, bitcast=True)
    signs = bits >> 31
    ordered_bits = bits ^ tl.where(signs != 0, 0xFFFFFFFF, 0x80000000)
    is_nan = bits & 0x7F800000 == 0x7F800000 and bits & 0x007FFFFF != 0
    is_zero = bits & 0x7FFFFFFF == 0
    ranks = tl.where(
        is_nan,
        0xFFFFFFFF,
        tl.where(is_zero, 0x80000000, ordered_bits),
    )
    reverse_offsets = 0xFFFFFFFF - offsets.to(tl.uint32)
    packed = (ranks.to(tl.uint64) << 32) | reverse_offsets.to(tl.uint64)

    winners = tl.max(packed, axis=0)
    tl.store(packed_ptr + program_id, winners.to(tl.int64, bitcast=True))


@triton.jit
def _finalize_max_index_kernel(
    input_ptr,
    packed_ptr,
    value_ptr,
    index_ptr,
    n_parts,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_parts
    packed = tl.load(packed_ptr + offsets, mask=mask, other=0)
    winner = tl.max(packed.to(tl.uint64, bitcast=True), axis=0)
    index = (winner & 0xFFFFFFFF).to(tl.int64)
    value = tl.load(input_ptr + index)

    tl.store(value_ptr, value)
    tl.store(index_ptr, index)


def build(context):
    def max(input, dim, keepdim=False):
        if input.ndim != 1 or dim != 0:
            raise ValueError("this kernel is specialized for the supplied 1-dimensional profile")

        n_elements = input.shape[0]
        block_size = 4096
        n_parts = (n_elements + block_size - 1) // block_size
        packed = torch.empty(
            (n_parts,), dtype=torch.int64, device=input.device
        )
        output_shape = [1] if keepdim else []
        values = torch.empty(output_shape, dtype=input.dtype, device=input.device)
        indices = torch.empty(output_shape, dtype=torch.int64, device=input.device)

        _block_max_index_kernel[(n_parts,)](
            input,
            packed,
            n_elements,
            BLOCK_SIZE=block_size,
            num_warps=8,
        )
        _finalize_max_index_kernel[(1,)](
            input,
            packed,
            values,
            indices,
            n_parts,
            BLOCK_SIZE=256,
            num_warps=2,
        )
        return values, indices

    return max
