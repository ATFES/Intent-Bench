from __future__ import annotations

import torch
import triton
import triton.language as tl


BLOCK_SIZE = 4096
MAX_STATS = 256


@triton.jit
def _block_stats_kernel(
    input_ptr,
    mean_ptr,
    m2_ptr,
    count_ptr,
    n_elements,
    EVEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK + tl.arange(0, BLOCK)

    if EVEN:
        values = tl.load(input_ptr + offsets)
        block_count = tl.full([], BLOCK, tl.float32)
    else:
        valid = offsets < n_elements
        values = tl.load(input_ptr + offsets, mask=valid, other=0.0)
        remaining = n_elements - block_id * BLOCK
        block_count = tl.minimum(remaining, BLOCK).to(tl.float32)

    block_mean = tl.sum(values, axis=0) / block_count
    if not EVEN:
        values = tl.where(valid, values - block_mean, 0.0)
    else:
        values = values - block_mean
    block_m2 = tl.sum(values * values, axis=0)

    tl.store(mean_ptr + block_id, block_mean)
    tl.store(m2_ptr + block_id, block_m2)
    tl.store(count_ptr + block_id, block_count)


@triton.jit
def _combine_stats_kernel(
    mean_ptr,
    m2_ptr,
    count_ptr,
    output_ptr,
    n_stats,
    correction,
    MAX_BLOCKS: tl.constexpr,
):
    stat_offsets = tl.arange(0, MAX_BLOCKS)
    stat_valid = stat_offsets < n_stats

    counts = tl.load(count_ptr + stat_offsets, mask=stat_valid, other=0.0)
    means = tl.load(mean_ptr + stat_offsets, mask=stat_valid, other=0.0)
    m2_values = tl.load(m2_ptr + stat_offsets, mask=stat_valid, other=0.0)

    total_count = tl.sum(counts, axis=0)
    global_mean = tl.sum(means * counts, axis=0) / total_count
    mean_shift = means - global_mean
    global_m2 = tl.sum(
        m2_values + counts * mean_shift * mean_shift,
        axis=0,
    )
    denominator = tl.maximum(total_count - correction, 0.0)
    variance = tl.where(denominator > 0.0, global_m2 / denominator, 0.0)
    variance = tl.maximum(variance, 0.0)
    standard_deviation = tl.sqrt_rn(variance)

    tl.store(output_ptr, standard_deviation)


def build(context):
    output = torch.empty((), dtype=torch.float32, device="cuda")
    means = torch.empty(MAX_STATS, dtype=torch.float32, device="cuda")
    m2_values = torch.empty(MAX_STATS, dtype=torch.float32, device="cuda")
    counts = torch.empty(MAX_STATS, dtype=torch.float32, device="cuda")

    def std(
        input: torch.Tensor,
        dim=None,
        *,
        correction: int = 1,
        keepdim: bool = False,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_elements = input.numel()
        n_stats = triton.cdiv(n_elements, BLOCK_SIZE)
        result = output if out is None else out

        _block_stats_kernel[(n_stats,)](
            input,
            means,
            m2_values,
            counts,
            n_elements,
            EVEN=(n_elements % BLOCK_SIZE == 0),
            BLOCK=BLOCK_SIZE,
            num_warps=8,
        )
        _combine_stats_kernel[(1,)](
            means,
            m2_values,
            counts,
            result,
            n_stats,
            float(correction),
            MAX_BLOCKS=MAX_STATS,
            num_warps=4,
        )
        return result

    return std
