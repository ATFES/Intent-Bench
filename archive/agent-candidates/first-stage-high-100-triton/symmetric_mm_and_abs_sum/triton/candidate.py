import torch
import triton
import triton.language as tl


@triton.jit
def _symmetric_tiles(
    a_ptr,
    c_ptr,
    partial_ptr,
    n,
    k_size,
    num_blocks,
    alpha,
    beta,
    stride_am,
    stride_ak,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Only the upper block triangle is needed: A @ A.T is symmetric.  The
    # lower-triangle programs return before touching the inputs.
    if pid_m > pid_n:
        return

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    row_valid = offs_m < n
    col_valid = offs_n < n

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k_size, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=row_valid[:, None] & (offs_k[None, :] < k_size),
            other=0.0,
        )
        # Load the source rows contiguously, then transpose them into the
        # logical right-hand operand.  This keeps the global loads coalesced.
        b_rows = tl.load(
            a_ptr + offs_n[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=col_valid[:, None] & (offs_k[None, :] < k_size),
            other=0.0,
        )
        b = tl.trans(b_rows)
        acc = tl.dot(a, b, acc=acc, input_precision="ieee")

    ij_valid = row_valid[:, None] & col_valid[None, :]
    ji_valid = col_valid[:, None] & row_valid[None, :]
    c_ij = tl.load(
        c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        mask=ij_valid,
        other=0.0,
    ).to(tl.float32)
    c_ji = tl.load(
        c_ptr + offs_n[:, None] * stride_cm + offs_m[None, :] * stride_cn,
        mask=ji_valid,
        other=0.0,
    ).to(tl.float32)

    value_ij = acc * alpha + c_ij * beta
    value_ji = tl.trans(acc) * alpha + c_ji * beta

    # On an off-diagonal block, both orientations are included.  On a
    # diagonal block, use the upper half of value_ij and lower half of its
    # transpose so that diagonal entries occur once and off-diagonal entries
    # occur twice.
    is_off_diagonal = pid_m < pid_n
    upper_ij = ij_valid & (is_off_diagonal | (offs_m[:, None] <= offs_n[None, :]))
    lower_ji = ji_valid & (is_off_diagonal | (offs_n[:, None] > offs_m[None, :]))
    abs_ij = tl.where(upper_ij, tl.abs(value_ij), 0.0)
    abs_ji = tl.where(lower_ji, tl.abs(value_ji), 0.0)
    tile_sum = tl.sum(tl.sum(abs_ij, axis=0), axis=0)
    tile_sum += tl.sum(tl.sum(abs_ji, axis=0), axis=0)

    partial_ptrs = partial_ptr + pid_m * num_blocks + pid_n
    tl.store(partial_ptrs, tile_sum)


@triton.jit
def _reduce_partials(partial_ptr, out_ptr, NUM_BLOCKS: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    rows = offsets // NUM_BLOCKS
    cols = offsets % NUM_BLOCKS
    mask = rows <= cols
    values = tl.load(partial_ptr + offsets, mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    tl.store(out_ptr, total)


def build(context):
    def wrapper(A, C, alpha, beta):
        n = A.shape[0]
        block_m = 32
        block_n = 32
        num_blocks = triton.cdiv(n, block_m)

        partials = torch.empty(
            (num_blocks, num_blocks), device=A.device, dtype=torch.float32
        )
        out = torch.empty((), device=A.device, dtype=torch.float32)

        _symmetric_tiles[(num_blocks, num_blocks)](
            A,
            C,
            partials,
            n,
            A.shape[1],
            num_blocks,
            alpha,
            beta,
            A.stride(0),
            A.stride(1),
            C.stride(0),
            C.stride(1),
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=32,
            num_warps=4,
            num_stages=3,
        )
        _reduce_partials[(1,)](
            partials,
            out,
            NUM_BLOCKS=num_blocks,
            BLOCK=num_blocks * num_blocks,
            num_warps=4,
        )
        return out

    return wrapper
