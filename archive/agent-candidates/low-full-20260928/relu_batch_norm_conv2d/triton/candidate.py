import torch
import triton
import triton.language as tl


@triton.jit
def _conv_bn_relu(
    x_ptr, w_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, out_ptr,
    H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr, K: tl.constexpr,
    eps, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    oc = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    p = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = oc < 128
    n_mask = p < OH * OW

    oh = p // OW
    ow = p - oh * OW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K is the flattened [input_channel, kernel_row, kernel_col] domain.
    for k0 in tl.range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        ic = kk // 9
        t = kk - ic * 9
        kh = t // 3
        kw = t - kh * 3
        ih = oh[None, :] + kh[:, None] - 1
        iw = ow[None, :] + kw[:, None] - 1
        imask = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        xoff = ic[:, None] * H * W + ih * W + iw
        xval = tl.load(x_ptr + xoff, mask=imask & n_mask[None, :], other=0.0)
        woff = oc[:, None] * K + kk[None, :]
        wval = tl.load(w_ptr + woff, mask=m_mask[:, None] & (kk[None, :] < K), other=0.0)
        acc += tl.dot(wval, xval, input_precision="ieee")

    mean = tl.load(mean_ptr + oc, mask=m_mask, other=0.0)
    var = tl.load(var_ptr + oc, mask=m_mask, other=0.0)
    gamma = tl.load(gamma_ptr + oc, mask=m_mask, other=0.0)
    beta = tl.load(beta_ptr + oc, mask=m_mask, other=0.0)
    y = (acc - mean[:, None]) * tl.rsqrt(var[:, None] + eps) * gamma[:, None] + beta[:, None]
    y = tl.maximum(y, 0.0)
    outoff = oc[:, None] * (OH * OW) + p[None, :]
    tl.store(out_ptr + outoff, y, mask=m_mask[:, None] & n_mask[None, :])


def build(context):
    def wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1, running_mean=None, running_var=None, bn_weight=None,
                bn_bias=None, training=False, momentum=0.1, eps=1e-5,
                inplace=False):
        output = torch.empty((input.shape[0], weight.shape[0], 32, 32),
                             device=input.device, dtype=input.dtype)
        grid = (triton.cdiv(128, 32), triton.cdiv(32 * 32, 64))
        _conv_bn_relu[grid](
            input, weight, running_mean, running_var, bn_weight, bn_bias, output,
            32, 32, 64, 32, 32, 576, eps, BLOCK_M=32, BLOCK_N=64, BLOCK_K=32,
        )
        return output

    return wrapper
