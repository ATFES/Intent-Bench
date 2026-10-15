import torch
import triton
import triton.language as tl


@triton.jit
def _log_softmax_linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_N)
    n_mask = offs_n < N

    # Invalid lanes (only possible when N is not a power of two) stay at -inf
    # so they cannot affect the max or exp-sum reductions.
    z = tl.where(n_mask, 0.0, -float("inf")).to(tl.float32)

    for k in tl.range(0, K, BLOCK_K, num_stages=2):
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        x = tl.load(x_ptr + row * K + offs_k, mask=k_mask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        z += tl.sum(w * x[None, :], axis=1)

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)
        z += bias

    max_z = tl.max(z, axis=0)
    shifted = z - max_z
    log_sum = tl.log(tl.sum(tl.exp(shifted), axis=0))

    result = shifted - log_sum
    tl.store(out_ptr + row * N + offs_n, result.to(out_ptr.dtype.element_ty), mask=n_mask)


def build(context):
    def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
        if dim < 0:
            dim += input.dim()
        if dim != input.dim() - 1:
            raise ValueError("log_softmax_linear currently supports softmax over the last dimension")

        in_features = int(weight.shape[1])
        out_features = int(weight.shape[0])
        if input.shape[-1] != in_features:
            raise ValueError("input and weight shapes are not compatible")

        out_dtype = input.dtype if dtype is None else dtype
        out = torch.empty(
            (*input.shape[:-1], out_features),
            device=input.device,
            dtype=out_dtype,
        )

        rows = input.numel() // in_features
        block_n = 1
        while block_n < out_features:
            block_n <<= 1
        block_k = 32

        _log_softmax_linear_kernel[(rows,)](
            input,
            weight,
            bias if bias is not None else weight,
            out,
            N=out_features,
            K=in_features,
            HAS_BIAS=bias is not None,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=8,
        )

        return out

    return log_softmax_linear
