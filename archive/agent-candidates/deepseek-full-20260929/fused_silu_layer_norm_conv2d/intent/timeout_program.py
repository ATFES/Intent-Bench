import torch
import intent
import intent.language as I


@intent.kernel
def conv2d_kernel(
    x: I.In[I.f32, ("N", "CIN", "H", "W")],
    weight: I.In[I.f32, ("COUT", "CIN", "KH", "KW")],
    bias: I.In[I.f32, ("COUT",)],
    y: I.Out[I.f32, ("N", "COUT", "OH", "OW")],
    has_bias: I.bool,
    conv_stride: I.i64,
    conv_padding: I.i64,
    conv_dilation: I.i64,
):
    N, CIN, H_in, W_in = x.shape
    COUT, CIN_W, KH, KW = weight.shape
    OH, OW = y.shape[2], y.shape[3]

    stride_i = I.cast(conv_stride, I.index)
    pad_i = I.cast(conv_padding, I.index)
    dil_i = I.cast(conv_dilation, I.index)

    ci_dom = I.domain(0, CIN)
    kh_dom = I.domain(0, KH)
    kw_dom = I.domain(0, KW)

    ci_i = I.reshape(I.indices(ci_dom), (CIN, 1, 1))
    kh_i = I.reshape(I.indices(kh_dom), (1, KH, 1))
    kw_i = I.reshape(I.indices(kw_dom), (1, 1, KW))

    for n in I.parallel(I.domain(0, N)):
        for co in I.parallel(I.domain(0, COUT)):
            for oh in I.parallel(I.domain(0, OH)):
                for ow in I.parallel(I.domain(0, OW)):
                    ih = oh * stride_i + kh_i * dil_i - pad_i
                    iw = ow * stride_i + kw_i * dil_i - pad_i

                    xv = x[n, ci_i, ih, iw]
                    wv = weight[co, ci_i, kh_i, kw_i]
                    acc = I.reduce.sum(xv * wv, axis=(0, 1, 2), acc_dtype=I.f32)

                    if has_bias:
                        y[n, co, oh, ow] = acc + bias[co]
                    else:
                        y[n, co, oh, ow] = acc


@intent.kernel
def spatial_stats_kernel(
    conv: I.In[I.f32, ("N", "COUT", "OH", "OW")],
    sum_part: I.Out[I.f32, ("N", "COUT")],
    sumsq_part: I.Out[I.f32, ("N", "COUT")],
):
    N, COUT, OH, OW = conv.shape
    oh_dom = I.domain(0, OH)
    ow_dom = I.domain(0, OW)

    for n in I.parallel(I.domain(0, N)):
        for co in I.parallel(I.domain(0, COUT)):
            vals = conv[n, co, oh_dom, ow_dom]
            s = I.reduce.sum(vals, axis=(0, 1), acc_dtype=I.f64)
            q = I.reduce.sum(vals * vals, axis=(0, 1), acc_dtype=I.f64)
            sum_part[n, co] = I.cast(s, I.f32)
            sumsq_part[n, co] = I.cast(q, I.f32)


@intent.kernel
def final_stats_kernel(
    sum_part: I.In[I.f32, ("N", "COUT")],
    sumsq_part: I.In[I.f32, ("N", "COUT")],
    mean: I.Out[I.f32, ("N",)],
    rstd: I.Out[I.f32, ("N",)],
    count: I.i64,
    eps: I.f32,
):
    N, COUT = sum_part.shape
    co_dom = I.domain(0, COUT)
    count_f = I.cast(count, I.f64)
    eps_f = I.cast(eps, I.f64)

    for n in I.parallel(I.domain(0, N)):
        s = I.reduce.sum(sum_part[n, co_dom], axis=0, acc_dtype=I.f64)
        q = I.reduce.sum(sumsq_part[n, co_dom], axis=0, acc_dtype=I.f64)

        mean_n = s / count_f
        var_n = q / count_f - mean_n * mean_n
        var_n = I.maximum(var_n, I.cast(0.0, I.f64))
        rstd_n = I.rsqrt(var_n + eps_f)

        mean[n] = I.cast(mean_n, I.f32)
        rstd[n] = I.cast(rstd_n, I.f32)


@intent.kernel
def norm_silu_kernel(
    conv: I.In[I.f32, ("N", "COUT", "OH", "OW")],
    mean: I.In[I.f32, ("N",)],
    rstd: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("N", "COUT", "OH", "OW")],
):
    N, COUT, OH, OW = conv.shape

    for n in I.parallel(I.domain(0, N)):
        for co in I.parallel(I.domain(0, COUT)):
            for oh in I.parallel(I.domain(0, OH)):
                for ow in I.parallel(I.domain(0, OW)):
                    z = (conv[n, co, oh, ow] - mean[n]) * rstd[n]
                    out[n, co, oh, ow] = z * I.sigmoid(z)


def build(context):
    conv_artifact = context.compile("conv2d", conv2d_kernel, constexprs={})
    spatial_stats_artifact = context.compile(
        "spatial_stats", spatial_stats_kernel, constexprs={}
    )
    final_stats_artifact = context.compile(
        "final_stats", final_stats_kernel, constexprs={}
    )
    norm_silu_artifact = context.compile(
        "norm_silu", norm_silu_kernel, constexprs={}
    )

    def fused_silu_layer_norm_conv2d(
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
        if conv_groups != 1:
            raise ValueError("conv_groups != 1 is not supported by this specialization")

        N, CIN, H, W = x.shape
        COUT, CIN_W, KH, KW = conv_weight.shape
        if CIN_W != CIN:
            raise ValueError("conv_weight input channels do not match x")

        OH = (H + 2 * conv_padding - conv_dilation * (KH - 1) - 1) // conv_stride + 1
        OW = (W + 2 * conv_padding - conv_dilation * (KW - 1) - 1) // conv_stride + 1

        y = torch.empty((N, COUT, OH, OW), device=x.device, dtype=torch.float32)

        has_bias = conv_bias is not None
        if has_bias:
            bias_in = conv_bias
        else:
            bias_in = torch.empty((COUT,), device=x.device, dtype=torch.float32)

        conv_artifact(
            x,
            conv_weight,
            bias_in,
            y,
            has_bias,
            int(conv_stride),
            int(conv_padding),
            int(conv_dilation),
        )

        sum_part = torch.empty((N, COUT), device=x.device, dtype=torch.float32)
        sumsq_part = torch.empty((N, COUT), device=x.device, dtype=torch.float32)
        spatial_stats_artifact(y, sum_part, sumsq_part)

        mean = torch.empty((N,), device=x.device, dtype=torch.float32)
        rstd = torch.empty((N,), device=x.device, dtype=torch.float32)
        final_stats_artifact(
            sum_part,
            sumsq_part,
            mean,
            rstd,
            int(COUT * OH * OW),
            float(ln_eps),
        )

        out = torch.empty((N, COUT, OH, OW), device=x.device, dtype=torch.float32)
        norm_silu_artifact(y, mean, rstd, out)
        return out

    return fused_silu_layer_norm_conv2d
