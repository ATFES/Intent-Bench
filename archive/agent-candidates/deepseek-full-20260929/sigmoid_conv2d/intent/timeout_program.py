import torch
import intent
import intent.language as I


def _as_pair(value, name):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or a pair")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _padding_pair(padding):
    if padding == "valid":
        return 0, 0
    if isinstance(padding, str):
        raise NotImplementedError("padding string is not supported by this specialization")
    return _as_pair(padding, "padding")


@intent.kernel
def _sigmoid_conv2d_kernel(
    input: I.In[I.f32, (1, 64, 64, 64)],
    weight: I.In[I.f32, (128, 64, 3, 3)],
    bias: I.In[I.f32, (128,)],
    out: I.Out[I.f32, (1, 128, 62, 62)],
    HAS_BIAS: I.Constexpr[bool],
):
    N, C, H, W = input.shape
    OC, C2, KH, KW = weight.shape
    OH = H - KH + 1
    OW = W - KW + 1

    M = N * OH * OW
    K = C * KH * KW

    m_d = I.domain(0, M)
    k_d = I.domain(0, K)
    n_d = I.domain(0, N)
    oh_d = I.domain(0, OH)
    ow_d = I.domain(0, OW)
    oc_d = I.domain(0, OC)

    # Implicit GEMM: flatten output pixels (n, oh, ow) into rows and
    # channel/kernel taps (c, kh, kw) into columns.
    m_idx = I.reshape(I.indices(m_d), (M, 1))
    k_idx = I.reshape(I.indices(k_d), (1, K))

    n_coord = m_idx // (OH * OW)
    oh_coord = (m_idx // OW) % OH
    ow_coord = m_idx % OW

    c_coord = k_idx // (KH * KW)
    kh_coord = (k_idx // KW) % KH
    kw_coord = k_idx % KW

    in_h = oh_coord + kh_coord
    in_w = ow_coord + kw_coord
    lhs = input[n_coord, c_coord, in_h, in_w]

    kk_idx = I.reshape(I.indices(k_d), (K, 1))
    oc_idx = I.reshape(I.indices(oc_d), (1, OC))
    wc_coord = kk_idx // (KH * KW)
    wkh_coord = (kk_idx // KW) % KH
    wkw_coord = kk_idx % KW
    rhs = weight[oc_idx, wc_coord, wkh_coord, wkw_coord]

    # lhs: [M, K], rhs: [K, OC] -> [M, OC]
    flat_acc = I.matmul(lhs, rhs, acc_dtype=I.f32)
    acc = I.reshape(flat_acc, (N, OH, OW, OC))
    acc_t = I.transpose(acc, (0, 3, 1, 2))

    if HAS_BIAS:
        bias_idx = I.reshape(I.indices(oc_d), (1, OC, 1, 1))
        y = I.sigmoid(acc_t + bias[bias_idx])
    else:
        y = I.sigmoid(acc_t)

    out_n = I.reshape(I.indices(n_d), (N, 1, 1, 1))
    out_oc = I.reshape(I.indices(oc_d), (1, OC, 1, 1))
    out_oh = I.reshape(I.indices(oh_d), (1, 1, OH, 1))
    out_ow = I.reshape(I.indices(ow_d), (1, 1, 1, OW))
    out[out_n, out_oc, out_oh, out_ow] = y


def build(context):
    bias_artifact = context.compile(
        "sigmoid_conv2d_bias",
        _sigmoid_conv2d_kernel,
        constexprs={"HAS_BIAS": True},
    )
    nobias_artifact = context.compile(
        "sigmoid_conv2d_nobias",
        _sigmoid_conv2d_kernel,
        constexprs={"HAS_BIAS": False},
    )

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
        stride_h, stride_w = _as_pair(stride, "stride")
        pad_h, pad_w = _padding_pair(padding)
        dilation_h, dilation_w = _as_pair(dilation, "dilation")
        if groups != 1:
            raise NotImplementedError("only groups=1 is supported")
        if (stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w) != (1, 1, 0, 0, 1, 1):
            raise NotImplementedError(
                "this specialization supports stride=1, padding=0, dilation=1"
            )

        N, C, H, W = input.shape
        OC, C2, KH, KW = weight.shape
        OH = (H + 2 * pad_h - dilation_h * (KH - 1) - 1) // stride_h + 1
        OW = (W + 2 * pad_w - dilation_w * (KW - 1) - 1) // stride_w + 1

        if out is None:
            out = torch.empty(
                (N, OC, OH, OW), dtype=input.dtype, device=input.device
            )

        if bias is None:
            # The kernel keeps the same ABI in both specializations; the
            # no-bias variant never reads this uninitialized view.
            dummy_bias = torch.empty((OC,), dtype=input.dtype, device=input.device)
            nobias_artifact(input, weight, dummy_bias, out)
        else:
            bias_artifact(input, weight, bias, out)

        return out

    return sigmoid_conv2d
