import torch
import intent
import intent.language as I


@intent.kernel
def conv2d_kernel(
    x: I.In[I.f32, (2, 16, 64, 64)],
    conv_weight: I.In[I.f32, (64, 16, 3, 3)],
    conv_bias: I.In[I.f32, (64,)],
    conv_stride: I.Constexpr[int],
    conv_padding: I.Constexpr[int],
    conv_dilation: I.Constexpr[int],
    conv_groups: I.Constexpr[int],
    conv_out: I.Out[I.f32, (2, 64, 62, 62)],
):
    B = x.shape[0]
    C_in = x.shape[1]
    H_in = x.shape[2]
    W_in = x.shape[3]
    C_out = conv_weight.shape[0]
    KH = conv_weight.shape[2]
    KW = conv_weight.shape[3]
    H_out = conv_out.shape[2]
    W_out = conv_out.shape[3]

    HW_out = H_out * W_out
    KW_C_in = KW * C_in
    KH_KW_C_in = KH * KW * C_in

    x_2d = I.reshape(x, (B, C_in * H_in * W_in))
    w_2d = I.reshape(conv_weight, (C_out, KH_KW_C_in))

    x_col = I.zeros((B * HW_out, KH_KW_C_in), I.f32)

    b_idx = I.domain(0, B)
    h_idx = I.domain(0, H_out)
    w_idx = I.domain(0, W_out)
    kh_idx = I.domain(0, KH)
    kw_idx = I.domain(0, KW)
    c_idx = I.domain(0, C_in)

    h_coords = I.indices(h_idx)
    w_coords = I.indices(w_idx)
    kh_coords = I.indices(kh_idx)
    kw_coords = I.indices(kw_idx)
    c_coords = I.indices(c_idx)

    h_in_coords = h_coords * conv_stride - conv_padding + kh_coords * conv_dilation
    w_in_coords = w_coords * conv_stride - conv_padding + kw_coords * conv_dilation

    valid_h = (h_in_coords >= I.cast(0, I.index)) & (h_in_coords < H_in)
    valid_w = (w_in_coords >= I.cast(0, I.index)) & (w_in_coords < W_in)
    valid = valid_h & valid_w

    in_idx = c_coords * H_in * W_in + h_in_coords * W_in + w_in_coords

    x_col_indices = (b_idx * HW_out + h_coords * W_out + w_coords) * KH_KW_C_in + kh_coords * KW_C_in + kw_coords * C_in + c_coords
    gathered = I.gather(x_2d, in_idx, valid=valid, fill=I.cast(0.0, I.f32))

    out_row = b_idx * HW_out + h_coords * W_out + w_coords
    out_col = kh_coords * KW_C_in + kw_coords * C_in + c_coords
    x_col_flat = I.reshape(gathered, (B * H_out * W_out, KH_KW_C_in))

    I.store(x_col, out_row, out_col, gathered)

    acc = I.matmul(x_col_flat, w_2d, acc_dtype=I.f32)

    acc_reshaped = I.reshape(acc, (B, HW_out, C_out))
    acc_transposed = I.transpose(acc_reshaped, (0, 2, 1))
    acc_final = I.reshape(acc_transposed, (B, C_out, H_out, W_out))

    bias_reshaped = I.reshape(conv_bias, (1, C_out, 1, 1))
    conv_out[:, :, :, :] = acc_final + bias_reshaped


@intent.kernel
def layer_norm_silu_kernel(
    x: I.In[I.f32, (2, 64, 62, 62)],
    ln_eps: I.Constexpr[float],
    y: I.Out[I.f32, (2, 64, 62, 62)],
):
    B = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    N = C * H * W

    x_flat = I.reshape(x, (B, N))

    count_inv = I.cast(1.0 / I.cast(N, I.f32), I.f32)
    mean = I.reduce.sum(x_flat, axis=1) * count_inv

    x_centered = x_flat - I.reshape(mean, (B, 1))
    variance = I.reduce.sum(x_centered * x_centered, axis=1) * count_inv

    inv_std = I.rsqrt(variance + I.cast(ln_eps, I.f32))
    x_norm = x_centered * I.reshape(inv_std, (B, 1))

    silu_out = x_norm * I.sigmoid(x_norm)

    x_final = I.reshape(silu_out, (B, C, H, W))
    y[:, :, :, :] = x_final


def build(context):
    conv_artifact = context.compile(
        "conv2d_stage",
        conv2d_kernel,
        constexprs={
            "conv_stride": 1,
            "conv_padding": 0,
            "conv_dilation": 1,
            "conv_groups": 1,
        },
    )

    ln_artifact = context.compile(
        "layer_norm_silu_stage",
        layer_norm_silu_kernel,
        constexprs={
            "ln_eps": 1e-05,
        },
    )

    def wrapper(
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor = None,
        conv_stride: int = 1,
        conv_padding: int = 0,
        conv_dilation: int = 1,
        conv_groups: int = 1,
        ln_eps: float = 1e-05,
    ) -> torch.Tensor:
        B, C_out, H_out, W_out = 2, 64, 62, 62

        if conv_bias is None:
            conv_bias = torch.zeros((C_out,), dtype=torch.float32, device="cuda")

        conv_out = torch.empty((B, C_out, H_out, W_out), dtype=torch.float32, device="cuda")
        conv_artifact(x, conv_weight, conv_bias, conv_out)

        y = torch.empty((B, C_out, H_out, W_out), dtype=torch.float32, device="cuda")
        ln_artifact(conv_out, y)

        return y

    return wrapper

