import torch
import intent
import intent.language as I


@intent.kernel
def _matmul(
    x: I.In[I.f32, ("M", "K")],
    w: I.In[I.f32, ("K", "N")],
    tmp: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, x.shape[0])
    cols = I.domain(0, w.shape[1])
    product = I.matmul(x[rows, :], w[:, cols], acc_dtype=I.f32)
    tmp[rows, cols] = product


@intent.kernel
def _activate(
    tmp: I.In[I.f32, ("M", "N")],
    weight2: I.In[I.f32, ("N",)],
    bias: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("M", "N")],
):
    rows = I.domain(0, tmp.shape[0])
    cols = I.domain(0, tmp.shape[1])
    z = tmp[rows, cols]
    sigmoid = I.sigmoid(z)
    activated = I.tanh(sigmoid)
    out[rows, cols] = activated * weight2[cols] + bias[cols]


def build(context):
    matmul = context.compile("combined_activation_matmul", _matmul)
    activate = context.compile("combined_activation_activate", _activate)

    def combined_activation(input, weight1, weight2, bias, *, out=None):
        m, k = input.shape
        n = weight1.shape[1]
        tmp = torch.empty((m, n), device=input.device, dtype=input.dtype)
        result = out if out is not None else torch.empty_like(tmp)
        matmul(input, weight1, tmp)
        activate(tmp, weight2, bias, result)
        return result

    return combined_activation
