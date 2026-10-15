import torch
import triton
import triton.language as tl


BLOCK_M = 32
BLOCK_N = 32
BLOCK_K = 16
BLOCK_OUT = 1024


@triton.jit
def conv_partial_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    y_ptr,
    sum_ptr,
    m2_ptr,
    count_ptr,
    N: tl.constexpr,
    C_IN: tl.constexpr,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    K: tl.constexpr,
    OH: tl.constexpr,
    OW: tl.constexpr,
    K_FLAT: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    SPATIAL_BLOCKS: tl.constexpr,
    C_BLOCKS: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    NUM_SPATIAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)

    spat_block = pid % SPATIAL_BLOCKS
    tmp = pid // SPATIAL_BLOCKS
    c_block = tmp % C_BLOCKS
    n = tmp // C_BLOCKS

    m_offs = c_block * BLOCK_M + tl.arange(0, BLOCK_M)
    spat_offs = spat_block * BLOCK_N + tl.arange(0, BLOCK_N)
    spat_mask = spat_offs < NUM_SPATIAL

    oh = spat_offs // OW
    ow = spat_offs % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K_FLAT, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        ic = k_offs // (R * S)
        rem = k_offs % (R * S)
        kh = rem // S
        kw = rem % S

        x_ptrs = (
            x_ptr
            + n * C_IN * H_IN * W_IN
            + ic[:, None] * (H_IN * W_IN)
            + (oh[None, :] + kh[:, None]) * W_IN
            + (ow[None, :] + kw[:, None])
        )
        xv = tl.load(x_ptrs, mask=spat_mask[None, :], other=0.0)

        w_ptrs = w_ptr + m_offs[:, None] * K_FLAT + k_offs[None, :]
        wv = tl.load(w_ptrs)

        acc = tl.dot(wv, xv, acc, input_precision="ieee")

    if HAS_BIAS:
        bias = tl.load(b_ptr + m_offs)
        acc = acc + bias[:, None]

    y_ptrs = (
        y_ptr
        + n * K * OH * OW
        + m_offs[:, None] * (OH * OW)
        + oh[None, :] * OW
        + ow[None, :]
    )
    valid = (m_offs[:, None] < K) & spat_mask[None, :]
    tl.store(y_ptrs, acc, mask=valid)

    masked = tl.where(valid, acc, 0.0)
    valid_f = tl.where(valid, 1.0, 0.0)
    local_count = tl.sum(valid_f)
    local_sum = tl.sum(masked)
    local_mean = local_sum / local_count
    centered = tl.where(valid, acc - local_mean, 0.0)
    local_m2 = tl.sum(centered * centered)

    slot = c_block * SPATIAL_BLOCKS + spat_block
    part_idx = n * NUM_PARTIALS + slot
    tl.store(sum_ptr + part_idx, local_sum)
    tl.store(m2_ptr + part_idx, local_m2)
    tl.store(count_ptr + part_idx, local_count)


@triton.jit
def stats_kernel(
    sum_ptr,
    m2_ptr,
    count_ptr,
    mean_ptr,
    rstd_ptr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_P: tl.constexpr,
    eps,
):
    n = tl.program_id(0)
    offs = tl.arange(0, BLOCK_P)
    mask = offs < NUM_PARTIALS
    base = n * NUM_PARTIALS + offs

    s = tl.load(sum_ptr + base, mask=mask, other=0.0)
    q = tl.load(m2_ptr + base, mask=mask, other=0.0)
    c = tl.load(count_ptr + base, mask=mask, other=1.0)

    total_sum = tl.sum(s)
    total_count = tl.sum(tl.where(mask, c, 0.0))
    mean = total_sum / total_count

    local_mean = s / c
    delta = local_mean - mean
    contribution = q + delta * delta * c
    total_m2 = tl.sum(tl.where(mask, contribution, 0.0))

    var = total_m2 / total_count
    var = tl.maximum(var, 0.0)
    rstd = tl.rsqrt(var + eps)

    tl.store(mean_ptr + n, mean)
    tl.store(rstd_ptr + n, rstd)


@triton.jit
def normalize_kernel(
    y_ptr,
    mean_ptr,
    rstd_ptr,
    TOTAL_PER_BATCH: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
):
    n = tl.program_id(0)
    block = tl.program_id(1)
    offs = block * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    mask = offs < TOTAL_PER_BATCH
    ptrs = y_ptr + n * TOTAL_PER_BATCH + offs

    y = tl.load(ptrs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + n)
    rstd = tl.load(rstd_ptr + n)
    z = (y - mean) * rstd
    out = z * tl.sigmoid(z)
    tl.store(ptrs, out, mask=mask)


def _next_power_of_two(n):
    p = 1
    while p < n:
        p *= 2
    return p


def build(context):
    def fused_silu_layer_norm_conv2d(
        x,
        weight,
        conv_weight,
        conv_bias=None,
        conv_stride=1,
        conv_padding=0,
        conv_dilation=1,
        conv_groups=1,
        ln_eps=1e-05,
    ):
        N, C_IN, H_IN, W_IN = x.shape
        K, Cg, R, S = conv_weight.shape

        OH = (H_IN + 2 * conv_padding - conv_dilation * (R - 1) - 1) // conv_stride + 1
        OW = (W_IN + 2 * conv_padding - conv_dilation * (S - 1) - 1) // conv_stride + 1
        NUM_SPATIAL = OH * OW
        K_FLAT = C_IN * R * S

        c_blocks = triton.cdiv(K, BLOCK_M)
        spatial_blocks = triton.cdiv(NUM_SPATIAL, BLOCK_N)
        num_partials = c_blocks * spatial_blocks
        total_per_batch = K * NUM_SPATIAL

        y = torch.empty((N, K, OH, OW), device=x.device, dtype=x.dtype)
        sum_buf = torch.empty((N, num_partials), device=x.device, dtype=torch.float32)
        m2_buf = torch.empty((N, num_partials), device=x.device, dtype=torch.float32)
        count_buf = torch.empty((N, num_partials), device=x.device, dtype=torch.float32)
        mean_buf = torch.empty((N,), device=x.device, dtype=torch.float32)
        rstd_buf = torch.empty((N,), device=x.device, dtype=torch.float32)

        bias_ptr = conv_bias if conv_bias is not None else x
        has_bias = conv_bias is not None

        grid = (N * c_blocks * spatial_blocks,)
        conv_partial_kernel[grid](
            x,
            conv_weight,
            bias_ptr,
            y,
            sum_buf,
            m2_buf,
            count_buf,
            N,
            C_IN,
            H_IN,
            W_IN,
            K,
            OH,
            OW,
            K_FLAT,
            R,
            S,
            spatial_blocks,
            c_blocks,
            num_partials,
            NUM_SPATIAL,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            HAS_BIAS=has_bias,
            num_warps=4,
        )

        block_p = _next_power_of_two(num_partials)
        stats_kernel[(N,)](
            sum_buf,
            m2_buf,
            count_buf,
            mean_buf,
            rstd_buf,
            NUM_PARTIALS=num_partials,
            BLOCK_P=block_p,
            eps=float(ln_eps),
            num_warps=4,
        )

        out_blocks = triton.cdiv(total_per_batch, BLOCK_OUT)
        normalize_kernel[(N, out_blocks)](
            y,
            mean_buf,
            rstd_buf,
            TOTAL_PER_BATCH=total_per_batch,
            BLOCK_OUT=BLOCK_OUT,
            num_warps=4,
        )

        return y

    return fused_silu_layer_norm_conv2d
