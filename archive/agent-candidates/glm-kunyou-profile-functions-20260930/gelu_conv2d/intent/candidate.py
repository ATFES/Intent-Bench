import torch
import intent
import intent.language as I


@intent.kernel
def gelu_conv2d_kernel(
    input: I.In[I.f32, (16, 16, 64, 64)],
    weight: I.In[I.f32, (32, 16, 3, 3)],
    bias: I.In[I.f32, (32,)],
    out: I.Out[I.f32, (16, 32, 64, 64)],
):
    stride, padding, dilation = 1, 1, 1
    N, C_in, H_in, W_in = 16, 16, 64, 64
    C_out, kH, kW = 32, 3, 3
    H_out = (H_in + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    W_out = (W_in + 2 * padding - dilation * (kW - 1) - 1) // stride + 1

    padded_H = H_in + 2 * padding
    padded_W = W_in + 2 * padding
    padded = I.full((N, C_in, padded_H, padded_W), I.cast(0.0, I.f32), I.f32)

    h_out_dom = I.domain(0, H_out)
    w_out_dom = I.domain(0, W_out)
    c_in_dom = I.domain(0, C_in)
    kh_dom = I.domain(0, kH)
    kw_dom = I.domain(0, kW)
    n_dom = I.domain(0, N)

    padded[n_dom, c_in_dom, h_out_dom + kh_dom, w_out_dom + kw_dom] = input[n_dom, c_in_dom, h_out_dom, w_out_dom]

    input_windows = I.reshape(
        padded[:, :, h_out_dom[0:H_out], w_out_dom[0:W_out]],
        (N, C_in, kH, kW, H_out, W_out)
    )
    input_reshaped = I.reshape(input_windows, (N * H_out * W_out, C_in * kH * kW))
    weight_reshaped = I.reshape(weight, (C_out, C_in * kH * kW))

    conv_out = I.matmul(input_reshaped, weight_reshaped, acc_dtype=I.f32)
    conv_out = I.reshape(conv_out, (N, H_out, W_out, C_out))
    conv_out = I.transpose(conv_out, (0, 3, 1, 2))

    inv_sqrt2 = I.cast(0.7071067811865476, I.f32)
    x_normalized = conv_out * inv_sqrt2
    erf_val = I.erf(x_normalized)
    cdf = I.cast(0.5, I.f32) * (I.cast(1.0, I.f32) + erf_val)
    out[:, :, :, :] = conv_out * cdf

    if bias is not None:
        bias_reshaped = I.reshape(bias, (1, C_out, 1, 1))
        out[:, :, :, :] = out + bias_reshaped


def build(context):
    artifact = context.compile(
        "gelu_conv2d_intent",
        gelu_conv2d_kernel,
        constexprs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1}
    )

    def gelu_conv2d(
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        approximate='none',
        out=None
    ):
        if (
            stride == 1
            and padding == 1
            and dilation == 1
            and groups == 1
            and input.shape == (16, 16, 64, 64)
            and weight.shape == (32, 16, 3, 3)
        ):
            N, C_in, H_in, W_in = input.shape
            C_out, _, kH, kW = weight.shape
            H_out = (H_in + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
            W_out = (W_in + 2 * padding - dilation * (kW - 1) - 1) // stride + 1

            if out is None:
                out = torch.empty((N, C_out, H_out, W_out), dtype=input.dtype, device=input.device)

            if bias is None:
                bias = torch.zeros(C_out, dtype=input.dtype, device=input.device)

            artifact(input, weight, bias, out)
            return out
        else:
            result = torch.nn.functional.conv2d(
                input, weight, bias=None, stride=stride, padding=padding,
                dilation=dilation, groups=groups
            )
            if approximate == 'none':
                result = result * 0.5 * (1 + torch.erf(result / 1.4142135623730951))
            else:
                sqrt2_pi = 0.7978845608028654
                x_cubed = result * result * result
                result = 0.5 * result * (1 + torch.tanh(sqrt2_pi * (result + 0.044715 * x_cubed)))

            if bias is not None:
                result = result + bias.view(1, -1, 1, 1)

            if out is not None:
                out.copy_(result)
                return out
            return result

    return gelu_conv2d

