import torch
import intent
import intent.language as I


@intent.kernel
def relu_conv2d_kernel(
    input: I.In[I.f32, (1, 3, 256, 256)],
    weight: I.In[I.f32, (64, 3, 3, 3)],
    stride_h: I.i64,
    stride_w: I.i64,
    pad_h: I.i64,
    pad_w: I.i64,
    dil_h: I.i64,
    dil_w: I.i64,
    H_out: I.i64,
    W_out: I.i64,
    out: I.Out[I.f32, (1, 64, 256, 256)],
):
    # Parallel over output elements
    n_domain = I.domain(0, 1)
    c_out_domain = I.domain(0, 64)
    h_out_domain = I.domain(0, H_out)
    w_out_domain = I.domain(0, W_out)

    for n in I.parallel(n_domain):
        for co in I.parallel(c_out_domain):
            for ho in I.parallel(h_out_domain):
                for wo in I.parallel(w_out_domain):
                    # Compute convolution sum
                    acc = I.cast(0.0, I.f32)
                    for ci in I.domain(0, 3):
                        for di in I.domain(0, 3):
                            for dj in I.domain(0, 3):
                                # Compute input coordinates with stride, padding, dilation
                                h_in = ho * I.cast(stride_h, I.index) - I.cast(pad_h, I.index) + di * I.cast(dil_h, I.index)
                                w_in = wo * I.cast(stride_w, I.index) - I.cast(pad_w, I.index) + dj * I.cast(dil_w, I.index)
                                # Check bounds
                                if h_in >= 0 and h_in < 256 and w_in >= 0 and w_in < 256:
                                    val = input[n, ci, h_in, w_in]
                                    wgt = weight[co, ci, di, dj]
                                    acc = acc + val * wgt
                    # Apply ReLU activation
                    out[n, co, ho, wo] = I.maximum(acc, I.cast(0.0, I.f32))


def build(context):
    artifact = context.compile("relu_conv2d", relu_conv2d_kernel)

    def wrapper(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, inplace=False):
        N, C_in, H_in, W_in = input.shape
        C_out, _, kH, kW = weight.shape

        # Handle stride/padding/dilation as int or tuple
        if isinstance(stride, int):
            stride_h = stride_w = stride
        else:
            stride_h, stride_w = stride

        if isinstance(padding, int):
            pad_h = pad_w = padding
        else:
            pad_h, pad_w = padding

        if isinstance(dilation, int):
            dil_h = dil_w = dilation
        else:
            dil_h, dil_w = dilation

        # Compute output shape
        H_out = (H_in + 2 * pad_h - dil_h * (kH - 1) - 1) // stride_h + 1
        W_out = (W_in + 2 * pad_w - dil_w * (kW - 1) - 1) // stride_w + 1

        # Allocate output
        out = torch.empty((N, C_out, H_out, W_out), dtype=torch.float32, device=input.device)

        # Launch kernel
        artifact(input, weight, stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, H_out, W_out, out)

        return out

    return wrapper

