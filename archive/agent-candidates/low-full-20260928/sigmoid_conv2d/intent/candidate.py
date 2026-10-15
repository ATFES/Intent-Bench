import torch
import intent
import intent.language as I


@intent.kernel
def _conv_sigmoid(
    x: I.In[I.f32, ("N", "CI", "IH", "IW")],
    w: I.In[I.f32, ("CO", "CI", 3, 3)],
    b: I.In[I.f32, ("CO",)],
    y: I.Out[I.f32, ("N", "CO", "OH", "OW")],
):
    n = I.domain(0, x.shape[0])
    co = I.domain(0, w.shape[0])
    oh = I.domain(0, y.shape[2])
    ow = I.domain(0, y.shape[3])
    k = I.domain(0, 9 * x.shape[1])

    ni = I.indices(n)
    coi = I.indices(co)
    ohi = I.indices(oh)
    owi = I.indices(ow)
    ki = I.indices(k)
    ci = ki // 9
    kh = (ki % 9) // 3
    kw = ki % 3

    # The singleton axes make the output and reduction coordinates explicit.
    n5 = I.reshape(ni, (x.shape[0], 1, 1, 1, 1))
    co5 = I.reshape(coi, (1, w.shape[0], 1, 1, 1))
    oh5 = I.reshape(ohi, (1, 1, y.shape[2], 1, 1))
    ow5 = I.reshape(owi, (1, 1, 1, y.shape[3], 1))
    ci5 = I.reshape(ci, (1, 1, 1, 1, 1, x.shape[1]))
    kh5 = I.reshape(kh, (1, 1, 1, 1, 1, 1, 3))
    kw5 = I.reshape(kw, (1, 1, 1, 1, 1, 1, 1, 3))

    # Flattened K is retained as the final logical reduction axis.
    ci_k = I.reshape(ci, (1, 1, 1, 1, 9 * x.shape[1]))
    kh_k = I.reshape(kh, (1, 1, 1, 1, 9 * x.shape[1]))
    kw_k = I.reshape(kw, (1, 1, 1, 1, 9 * x.shape[1]))
    xvals = x[n5, ci_k, oh5 + kh_k, ow5 + kw_k]
    wvals = w[co5, ci_k, kh_k, kw_k]
    products = xvals * wvals
    acc = I.reduce.sum(products, axis=4, acc_dtype=I.f32)
    acc = acc + b[coi]
    one = I.cast(1.0, I.f32)
    y[n, co, oh, ow] = one / (one + I.exp(-acc))


def build(context):
    kernel = context.compile("sigmoid_conv2d", _conv_sigmoid)

    def sigmoid_conv2d(input, weight, bias=None, stride=1, padding=0,
                       dilation=1, groups=1, out=None):
        if isinstance(stride, tuple):
            sh, sw = stride
        else:
            sh = sw = stride
        if isinstance(dilation, tuple):
            dh, dw = dilation
        else:
            dh = dw = dilation
        if isinstance(padding, str):
            if padding == "valid":
                ph = pw = 0
            elif padding == "same":
                ph = (weight.shape[2] - 1) * dh // 2
                pw = (weight.shape[3] - 1) * dw // 2
            else:
                raise ValueError("unsupported padding")
        elif isinstance(padding, tuple):
            ph, pw = padding
        else:
            ph = pw = padding
        oh = (input.shape[2] + 2 * ph - dh * (weight.shape[2] - 1) - 1) // sh + 1
        ow = (input.shape[3] + 2 * pw - dw * (weight.shape[3] - 1) - 1) // sw + 1
        result = out if out is not None else torch.empty(
            (input.shape[0], weight.shape[0], oh, ow), device=input.device,
            dtype=input.dtype)
        if bias is None:
            bias = torch.zeros((weight.shape[0],), device=input.device, dtype=input.dtype)
        kernel(input, weight, bias, result)
        return result

    return sigmoid_conv2d
