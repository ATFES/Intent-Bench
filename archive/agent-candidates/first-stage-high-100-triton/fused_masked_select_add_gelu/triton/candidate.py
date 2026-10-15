import torch
import triton
import triton.language as tl


N_ELEMENTS = 1048576
BLOCK = 1024
N_BLOCKS = N_ELEMENTS // BLOCK
OUTPUT_N = 524236


@triton.jit
def _count_mask_kernel(mask_ptr, counts_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    mask = tl.load(mask_ptr + offsets, mask=valid, other=0)
    count = tl.sum(mask.to(tl.int32), axis=0)
    tl.store(counts_ptr + pid, count)


@triton.jit
def _prefix_counts_kernel(counts_ptr, offsets_ptr, N_BLOCKS: tl.constexpr):
    offsets = tl.arange(0, N_BLOCKS)
    counts = tl.load(counts_ptr + offsets)
    prefix = tl.cumsum(counts, axis=0) - counts
    tl.store(offsets_ptr + offsets, prefix)


@triton.jit
def _compact_exact_kernel(
    input_ptr,
    mask_ptr,
    block_offsets_ptr,
    output_ptr,
    n_elements,
    other,
    alpha,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    selected = tl.load(mask_ptr + offsets, mask=valid, other=0)
    values = tl.load(input_ptr + offsets, mask=valid, other=0.0)

    local_rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
    block_base = tl.load(block_offsets_ptr + pid)
    output_offsets = block_base + local_rank

    z = values + alpha * other
    z = 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))
    tl.store(output_ptr + output_offsets, z, mask=selected)


@triton.jit
def _compact_tanh_kernel(
    input_ptr,
    mask_ptr,
    block_offsets_ptr,
    output_ptr,
    n_elements,
    other,
    alpha,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    valid = offsets < n_elements
    selected = tl.load(mask_ptr + offsets, mask=valid, other=0)
    values = tl.load(input_ptr + offsets, mask=valid, other=0.0)

    local_rank = tl.cumsum(selected.to(tl.int32), axis=0) - 1
    block_base = tl.load(block_offsets_ptr + pid)
    output_offsets = block_base + local_rank

    z = values + alpha * other
    inner = 0.7978845608028654 * (z + 0.044715 * z * z * z)
    z = 0.5 * z * (1.0 + tl.math.tanh(inner))
    tl.store(output_ptr + output_offsets, z, mask=selected)


def build(context):
    def wrapper(input, mask, other, *, alpha=1, approximate="none", out=None):
        if out is None:
            out = torch.empty(
                (OUTPUT_N,), device=input.device, dtype=input.dtype
            )

        counts = torch.empty((N_BLOCKS,), device=input.device, dtype=torch.int32)
        block_offsets = torch.empty(
            (N_BLOCKS,), device=input.device, dtype=torch.int32
        )

        grid = (N_BLOCKS,)
        _count_mask_kernel[grid](
            mask, counts, N_ELEMENTS, BLOCK=BLOCK
        )
        _prefix_counts_kernel[(1,)](
            counts, block_offsets, N_BLOCKS=N_BLOCKS
        )

        kernel = (
            _compact_tanh_kernel if approximate == "tanh" else _compact_exact_kernel
        )
        kernel[grid](
            input,
            mask,
            block_offsets,
            out,
            N_ELEMENTS,
            other,
            alpha,
            BLOCK=BLOCK,
        )
        return out

    return wrapper
