from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rsqrt_gemm_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    m,
    n,
    k,
    a_stride_m,
    a_stride_k,
    b_stride_k,
    b_stride_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(m, BLOCK_M)
    num_pid_n = tl.cdiv(n, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offsets_k = tl.arange(0, BLOCK_K)
    mask_m = offsets_m < m
    mask_n = offsets_n < n

    a_ptrs = a_ptr + offsets_m[:, None] * a_stride_m + offsets_k[None, :] * a_stride_k
    b_ptrs = b_ptr + offsets_k[:, None] * b_stride_k + offsets_n[None, :] * b_stride_n
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K:
        for _ in range(0, k, BLOCK_K):
            a_tile = tl.load(a_ptrs)
            b_tile = tl.load(b_ptrs)
            accumulator = tl.dot(
                a_tile,
                b_tile,
                accumulator,
                input_precision="tf32x3",
            )
            a_ptrs += BLOCK_K * a_stride_k
            b_ptrs += BLOCK_K * b_stride_k
    else:
        for block_start in range(0, k, BLOCK_K):
            mask_k = block_start + offsets_k < k
            a_tile = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            b_tile = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            accumulator = tl.dot(
                a_tile,
                b_tile,
                accumulator,
                input_precision="tf32x3",
            )
            a_ptrs += BLOCK_K * a_stride_k
            b_ptrs += BLOCK_K * b_stride_k

    output_offsets = offsets_m[:, None] * n + offsets_n[None, :]
    output_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(output_ptr + output_offsets, tl.rsqrt(accumulator), mask=output_mask)


@triton.jit
def _rsqrt_contract_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    output_count,
    contraction_count,
    output_sizes: tl.constexpr,
    output_radix: tl.constexpr,
    output_sources: tl.constexpr,
    output_strides: tl.constexpr,
    contraction_sizes: tl.constexpr,
    contraction_radix: tl.constexpr,
    a_contraction_strides: tl.constexpr,
    b_contraction_strides: tl.constexpr,
    BLOCK: tl.constexpr,
):
    output_flat = tl.program_id(axis=0) * BLOCK + tl.arange(0, BLOCK)
    output_mask = output_flat < output_count
    offsets_flat = output_flat.to(tl.int64)
    a_offset = tl.zeros((1,), dtype=tl.int64)
    b_offset = tl.zeros((1,), dtype=tl.int64)

    for dimension in tl.static_range(len(output_sizes)):
        index = (offsets_flat // output_radix[dimension]) % output_sizes[dimension]
        stride = output_strides[dimension]
        if output_sources[dimension] == 0:
            a_offset += index * stride
        else:
            b_offset += index * stride

    result = tl.zeros((BLOCK,), dtype=tl.float32)
    for contraction_flat in range(0, contraction_count):
        contraction_offsets = contraction_flat.to(tl.int64)
        a_contraction_offset = tl.zeros((1,), dtype=tl.int64)
        b_contraction_offset = tl.zeros((1,), dtype=tl.int64)
        for dimension in tl.static_range(len(contraction_sizes)):
            index = (contraction_offsets // contraction_radix[dimension]) % contraction_sizes[dimension]
            a_contraction_offset += index * a_contraction_strides[dimension]
            b_contraction_offset += index * b_contraction_strides[dimension]
        result += tl.load(a_ptr + a_offset + a_contraction_offset) * tl.load(
            b_ptr + b_offset + b_contraction_offset
        )

    tl.store(output_ptr + output_flat, tl.rsqrt(result), mask=output_mask)


def _normalize_dimensions(a, b, dims):
    if isinstance(dims, int):
        if dims < 0:
            raise ValueError("tensordot dimensions must be non-negative")
        a_dims = list(range(a.ndim - dims, a.ndim))
        b_dims = list(range(dims))
    else:
        if len(dims) != 2:
            raise ValueError("tensordot dimensions must be an int or a pair")
        first, second = dims
        a_dims = [first] if isinstance(first, int) else list(first)
        b_dims = [second] if isinstance(second, int) else list(second)
        if len(a_dims) != len(b_dims):
            raise ValueError("tensordot dimension lists must have equal length")
        a_dims = [dimension % a.ndim for dimension in a_dims]
        b_dims = [dimension % b.ndim for dimension in b_dims]

    if len(set(a_dims)) != len(a_dims) or len(set(b_dims)) != len(b_dims):
        raise ValueError("tensordot dimensions must be unique")
    if len(a_dims) != len(b_dims):
        raise ValueError("tensordot dimension lists must have equal length")
    for a_dim, b_dim in zip(a_dims, b_dims):
        if a.shape[a_dim] != b.shape[b_dim]:
            raise ValueError("tensordot contraction dimensions must have equal sizes")
    return a_dims, b_dims


def _radix(sizes):
    result = []
    multiplier = 1
    for size in reversed(sizes):
        result.append(multiplier)
        multiplier *= size
    return tuple(reversed(result))


def build(context):
    def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
        a_dimensions, b_dimensions = _normalize_dimensions(a, b, dims)

        if (
            a.ndim == 2
            and b.ndim == 2
            and a_dimensions == [0]
            and b_dimensions == [1]
        ):
            m = a.shape[0]
            n = b.shape[1]
            k = a.shape[1]
            output = torch.empty((m, n), device=a.device, dtype=torch.float32)
            block_m = 64
            block_n = 64
            block_k = 64
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            _rsqrt_gemm_kernel[grid](
                a,
                b,
                output,
                m,
                n,
                k,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                GROUP_SIZE_M=8,
                EVEN_K=(k % block_k == 0),
                num_warps=4,
                num_stages=3,
            )
            return output

        a_free_dimensions = [
            dimension for dimension in range(a.ndim) if dimension not in a_dimensions
        ]
        b_free_dimensions = [
            dimension for dimension in range(b.ndim) if dimension not in b_dimensions
        ]
        output_sizes = tuple(a.shape[d] for d in a_free_dimensions) + tuple(
            b.shape[d] for d in b_free_dimensions
        )
        output_sources = tuple(0 for _ in a_free_dimensions) + tuple(
            1 for _ in b_free_dimensions
        )
        output_strides = tuple(a.stride(d) for d in a_free_dimensions) + tuple(
            b.stride(d) for d in b_free_dimensions
        )
        contraction_sizes = tuple(a.shape[d] for d in a_dimensions)
        a_contraction_strides = tuple(a.stride(d) for d in a_dimensions)
        b_contraction_strides = tuple(b.stride(d) for d in b_dimensions)
        output_count = 1
        for size in output_sizes:
            output_count *= size
        contraction_count = 1
        for size in contraction_sizes:
            contraction_count *= size

        output = torch.empty(output_sizes, device=a.device, dtype=torch.float32)
        if output_count == 0:
            return output

        block = 256
        grid = (triton.cdiv(output_count, block),)
        _rsqrt_contract_kernel[grid](
            a,
            b,
            output,
            output_count,
            contraction_count,
            output_sizes=output_sizes,
            output_radix=_radix(output_sizes),
            output_sources=output_sources,
            output_strides=output_strides,
            contraction_sizes=contraction_sizes,
            contraction_radix=_radix(contraction_sizes),
            a_contraction_strides=a_contraction_strides,
            b_contraction_strides=b_contraction_strides,
            BLOCK=block,
            num_warps=4,
        )
        return output

    return tensordot_rsqrt
