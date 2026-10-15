import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C_PER_GROUP: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_H_FRONT: tl.constexpr,
    PAD_W_FRONT: tl.constexpr,
    DIL_H: tl.constexpr,
    DIL_W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    IN_S_B: tl.constexpr,
    IN_S_C: tl.constexpr,
    IN_S_H: tl.constexpr,
    IN_S_W: tl.constexpr,
    W_S_O: tl.constexpr,
    W_S_C: tl.constexpr,
    W_S_H: tl.constexpr,
    W_S_W: tl.constexpr,
    OUT_S_B: tl.constexpr,
    OUT_S_O: tl.constexpr,
    OUT_S_H: tl.constexpr,
    OUT_S_W: tl.constexpr,
    OUT_PER_GROUP: tl.constexpr,
    GROUPS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    PRECISION: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bg = tl.program_id(2)

    batch = pid_bg // GROUPS
    gid = pid_bg % GROUPS

    oc = gid * OUT_PER_GROUP + pid_m * BM + tl.arange(0, BM)
    m_valid = oc < (gid + 1) * OUT_PER_GROUP

    n = pid_n * BN + tl.arange(0, BN)
    n_valid = n < OUT_H * OUT_W
    oh = n // OUT_W
    ow = n % OUT_W

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    K = C_PER_GROUP * KH * KW
    for k_start in tl.static_range(0, K, BK):
        k = k_start + tl.arange(0, BK)
        k_valid = k < K

        ci = k // (KH * KW)
        rem = k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        a_ci = gid * C_PER_GROUP + ci[None, :]
        a_off = oc[:, None] * W_S_O + a_ci * W_S_C + kh[None, :] * W_S_H + kw[None, :] * W_S_W
        a_mask = m_valid[:, None] & k_valid[None, :]
        a = tl.load(weight_ptr + a_off, mask=a_mask, other=0.0)

        ih = oh[None, :] * STRIDE_H - PAD_H_FRONT + kh[:, None] * DIL_H
        iw = ow[None, :] * STRIDE_W - PAD_W_FRONT + kw[:, None] * DIL_W
        b_ci = gid * C_PER_GROUP + ci[:, None]
        b_off = batch * IN_S_B + b_ci * IN_S_C + ih * IN_S_H + iw * IN_S_W
        b_valid = k_valid[:, None] & n_valid[None, :] & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
        b = tl.load(input_ptr + b_off, mask=b_valid, other=0.0)

        acc = tl.dot(a, b, acc, input_precision=PRECISION)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + oc, mask=m_valid, other=0.0)
        acc = acc + bias[:, None]

    sig = tl.sigmoid(acc)

    out_off = batch * OUT_S_B + oc[:, None] * OUT_S_O + oh[None, :] * OUT_S_H + ow[None, :] * OUT_S_W
    out_mask = m_valid[:, None] & n_valid[None, :]
    tl.store(out_ptr + out_off, sig, mask=out_mask)


def _pair(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("expected a pair of ints")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def build(context):
    def sigmoid_conv2d(
        input,
        weight,
        bias=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        out=None,
    ):
        batch_size = input.shape[0]
        in_c = input.shape[1]
        in_h = input.shape[2]
        in_w = input.shape[3]

        out_per_group = weight.shape[0]
        c_per_group = weight.shape[1]
        kh = weight.shape[2]
        kw = weight.shape[3]

        stride_h, stride_w = _pair(stride)
        dil_h, dil_w = _pair(dilation)
        groups = int(groups)

        if isinstance(padding, str):
            if padding == "valid":
                pad_h_front = pad_h_back = 0
                pad_w_front = pad_w_back = 0
            elif padding == "same":
                out_h = (in_h + stride_h - 1) // stride_h
                out_w = (in_w + stride_w - 1) // stride_w
                total_h = max((out_h - 1) * stride_h + dil_h * (kh - 1) + 1 - in_h, 0)
                total_w = max((out_w - 1) * stride_w + dil_w * (kw - 1) + 1 - in_w, 0)
                pad_h_front = total_h // 2
                pad_h_back = total_h - pad_h_front
                pad_w_front = total_w // 2
                pad_w_back = total_w - pad_w_front
            else:
                raise ValueError("padding string must be 'valid' or 'same'")
        else:
            pad_h, pad_w = _pair(padding)
            pad_h_front = pad_h_back = pad_h
            pad_w_front = pad_w_back = pad_w

        out_h = (in_h + pad_h_front + pad_h_back - dil_h * (kh - 1) - 1) // stride_h + 1
        out_w = (in_w + pad_w_front + pad_w_back - dil_w * (kw - 1) - 1) // stride_w + 1

        out_c = out_per_group * groups

        if out is None:
            out = torch.empty((batch_size, out_c, out_h, out_w), device=input.device, dtype=input.dtype)

        in_s_b, in_s_c, in_s_h, in_s_w = input.stride()
        w_s_o, w_s_c, w_s_h, w_s_w = weight.stride()
        out_s_b, out_s_o, out_s_h, out_s_w = out.stride()

        has_bias = bias is not None
        bias_arg = bias if bias is not None else input

        BM = 64
        BN = 64
        BK = 64
        grid = (
            triton.cdiv(out_per_group, BM),
            triton.cdiv(out_h * out_w, BN),
            batch_size * groups,
        )

        _sigmoid_conv2d_kernel[grid](
            input,
            weight,
            bias_arg,
            out,
            C_PER_GROUP=c_per_group,
            H=in_h,
            W=in_w,
            KH=kh,
            KW=kw,
            STRIDE_H=stride_h,
            STRIDE_W=stride_w,
            PAD_H_FRONT=pad_h_front,
            PAD_W_FRONT=pad_w_front,
            DIL_H=dil_h,
            DIL_W=dil_w,
            OUT_H=out_h,
            OUT_W=out_w,
            IN_S_B=in_s_b,
            IN_S_C=in_s_c,
            IN_S_H=in_s_h,
            IN_S_W=in_s_w,
            W_S_O=w_s_o,
            W_S_C=w_s_c,
            W_S_H=w_s_h,
            W_S_W=w_s_w,
            OUT_S_B=out_s_b,
            OUT_S_O=out_s_o,
            OUT_S_H=out_s_h,
            OUT_S_W=out_s_w,
            OUT_PER_GROUP=out_per_group,
            GROUPS=groups,
            HAS_BIAS=has_bias,
            BM=BM,
            BN=BN,
            BK=BK,
            PRECISION="ieee",
            num_warps=8,
        )
        return out

    return sigmoid_conv2d
