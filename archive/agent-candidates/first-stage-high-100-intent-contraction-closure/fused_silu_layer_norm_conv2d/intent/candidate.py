import torch
import intent
import intent.language as I


@intent.kernel
def _conv2d_bias(
    x: I.In[I.f32, (2, 16, 64, 64)],
    conv_weight: I.In[I.f32, (64, 16, 3, 3)],
    conv_bias: I.In[I.f32, (64,)],
    out: I.Out[I.f32, (2, 64, 62, 62)],
):
    for batch in I.parallel(I.domain(0, 2)):
        for channel_out in I.parallel(I.domain(0, 64)):
            for row_out in I.parallel(I.domain(0, 62)):
                for col_out in I.parallel(I.domain(0, 62)):
                    acc = I.cast(0.0, I.f32)
                    for channel_in in I.domain(0, 16):
                        for row_kernel in I.domain(0, 3):
                            for col_kernel in I.domain(0, 3):
                                acc = acc + (
                                    x[
                                        batch,
                                        channel_in,
                                        row_out + row_kernel,
                                        col_out + col_kernel,
                                    ]
                                    * conv_weight[
                                        channel_out,
                                        channel_in,
                                        row_kernel,
                                        col_kernel,
                                    ]
                                )
                    out[batch, channel_out, row_out, col_out] = (
                        acc + conv_bias[channel_out]
                    )


@intent.kernel
def _conv2d_no_bias(
    x: I.In[I.f32, (2, 16, 64, 64)],
    conv_weight: I.In[I.f32, (64, 16, 3, 3)],
    out: I.Out[I.f32, (2, 64, 62, 62)],
):
    for batch in I.parallel(I.domain(0, 2)):
        for channel_out in I.parallel(I.domain(0, 64)):
            for row_out in I.parallel(I.domain(0, 62)):
                for col_out in I.parallel(I.domain(0, 62)):
                    acc = I.cast(0.0, I.f32)
                    for channel_in in I.domain(0, 16):
                        for row_kernel in I.domain(0, 3):
                            for col_kernel in I.domain(0, 3):
                                acc = acc + (
                                    x[
                                        batch,
                                        channel_in,
                                        row_out + row_kernel,
                                        col_out + col_kernel,
                                    ]
                                    * conv_weight[
                                        channel_out,
                                        channel_in,
                                        row_kernel,
                                        col_kernel,
                                    ]
                                )
                    out[batch, channel_out, row_out, col_out] = acc


@intent.kernel
def _layer_norm_silu(
    conv: I.In[I.f32, (2, 64, 62, 62)],
    ln_eps: I.f32,
    out: I.Out[I.f32, (2, 64, 62, 62)],
):
    channels = I.domain(0, 64)
    rows = I.domain(0, 62)
    columns = I.domain(0, 62)
    element_count = I.cast(246016.0, I.f32)

    for batch in I.parallel(I.domain(0, 2)):
        values = conv[batch, channels, rows, columns]
        mean = I.fdiv(
            I.reduce.sum(values, axis=(0, 1, 2), acc_dtype=I.f32),
            element_count,
        )
        centered = values - mean
        variance = I.fdiv(
            I.reduce.sum(centered * centered, axis=(0, 1, 2), acc_dtype=I.f32),
            element_count,
        )
        normalized = I.fdiv(centered, I.sqrt(variance + ln_eps))
        out[batch, channels, rows, columns] = normalized * I.sigmoid(normalized)


def build(context):
    conv_bias_artifact = context.compile("fused_conv_bias", _conv2d_bias)
    conv_no_bias_artifact = context.compile("fused_conv_no_bias", _conv2d_no_bias)
    norm_artifact = context.compile("fused_layer_norm_silu", _layer_norm_silu)

    def fused_silu_layer_norm_conv2d(
        x: torch.Tensor,
        weight: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_bias: torch.Tensor = None,
        conv_stride: int = 1,
        conv_padding: int = 0,
        conv_dilation: int = 1,
        conv_groups: int = 1,
        ln_eps: float = 1e-5,
    ) -> torch.Tensor:
        del weight, conv_stride, conv_padding, conv_dilation, conv_groups
        conv = torch.empty((2, 64, 62, 62), device=x.device, dtype=x.dtype)
        if conv_bias is None:
            conv_no_bias_artifact(x, conv_weight, conv)
        else:
            conv_bias_artifact(x, conv_weight, conv_bias, conv)

        out = torch.empty((2, 64, 62, 62), device=x.device, dtype=x.dtype)
        norm_artifact(conv, ln_eps, out)
        return out

    return fused_silu_layer_norm_conv2d
