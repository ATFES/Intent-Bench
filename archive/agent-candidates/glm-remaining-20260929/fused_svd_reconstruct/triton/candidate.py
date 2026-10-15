from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _zero_counter(counter_ptr):
    tl.store(counter_ptr, 0)


@triton.jit
def _one_sided_jacobi(
    input_ptr,
    output_ptr,
    counter_ptr,
    N: tl.constexpr,
    PAIRS_PER_ROUND: tl.constexpr,
    BASE: tl.constexpr,
    SWEEPS: tl.constexpr,
    ROUNDS: tl.constexpr,
):
    lane = tl.program_id(0)
    lanes = tl.arange(0, N)
    first_copy_row = 2 * lane
    second_copy_row = first_copy_row + 1

    first_input = input_ptr + first_copy_row * N + lanes
    second_input = input_ptr + second_copy_row * N + lanes
    first_output = output_ptr + first_copy_row * N + lanes
    second_output = output_ptr + second_copy_row * N + lanes
    tl.store(first_output, tl.load(first_input))
    tl.store(second_output, tl.load(second_input))

    arrival = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
    while arrival < PAIRS_PER_ROUND:
        arrival = tl.atomic_add(counter_ptr, 0, sem="acquire", scope="gpu")

    for step in tl.range(0, SWEEPS * ROUNDS):
        sweep = step // ROUNDS
        round_index = step % ROUNDS
        lane_offset = lane * 2
        first_column = (round_index + lane_offset) % BASE
        second_offset = round_index - lane_offset + BASE
        second_column = second_offset % BASE

        first = tl.load(output_ptr + first_column * N + lanes)
        first_norm = tl.sum(first * first, axis=0)
        second = tl.load(output_ptr + second_column * N + lanes)
        second_norm = tl.sum(second * second, axis=0)
        cross = tl.sum(first * second, axis=0)
        cross_abs = tl.abs(cross)
        scale = tl.sqrt(first_norm * second_norm)

        if (first_norm > 0.0) & (second_norm > 0.0) & (cross_abs > 1.0e-8 * scale):
            delta = (second_norm - first_norm) / (2.0 * cross)
            delta_abs = tl.abs(delta)
            tangent = tl.where(delta >= 0.0, 1.0, -1.0) / (
                delta_abs + tl.sqrt(1.0 + delta * delta)
            )
            cosine = 1.0 / tl.sqrt(1.0 + tangent * tangent)
            sine = cosine * tangent
            rotated_first = cosine * first - sine * second
            second = sine * first + cosine * second
            tl.store(output_ptr + first_column * N + lanes, rotated_first)
            tl.store(output_ptr + second_column * N + lanes, second)

        expected = (step + 1) * PAIRS_PER_ROUND
        arrival = tl.atomic_add(counter_ptr, 1, sem="acq_rel", scope="gpu")
        while arrival < expected:
            arrival = tl.atomic_add(counter_ptr, 0, sem="acquire", scope="gpu")


def build(context):
    def fused_svd_reconstruct(A: torch.Tensor) -> torch.Tensor:
        rotated = torch.empty_like(A)
        counter = torch.empty((1,), dtype=torch.int32, device=A.device)
        _zero_counter[(1,)](counter)
        _one_sided_jacobi[(32,)](
            A,
            rotated,
            counter,
            N=64,
            PAIRS_PER_ROUND=32,
            BASE=63,
            SWEEPS=12,
            ROUNDS=63,
            num_warps=2,
        )
        return rotated

    return fused_svd_reconstruct

