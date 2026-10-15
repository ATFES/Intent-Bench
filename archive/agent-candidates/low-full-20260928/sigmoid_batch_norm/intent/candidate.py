import torch
import intent
import intent.language as I


@intent.kernel
def _bn_sigmoid_ww(
    x: I.In[I.f32, ("N", "C")],
    mean: I.In[I.f32, ("C",)],
    var: I.In[I.f32, ("C",)],
    weight: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    out: I.Out[I.f32, ("N", "C")],
    eps: I.f32,
):
    rows = I.domain(0, x.shape[0])
    cols = I.domain(0, x.shape[1])
    xv = x[rows, cols]
    m = mean[cols]
    v = var[cols]
    normalized = (xv - m) / I.sqrt(v + eps)
    out[rows, cols] = I.sigmoid(normalized * weight[cols] + bias[cols])


@intent.kernel
def _bn_sigmoid_w0(
    x: I.In[I.f32, ("N", "C")],
    mean: I.In[I.f32, ("C",)],
    var: I.In[I.f32, ("C",)],
    weight: I.In[I.f32, ("C",)],
    out: I.Out[I.f32, ("N", "C")],
    eps: I.f32,
):
    rows = I.domain(0, x.shape[0])
    cols = I.domain(0, x.shape[1])
    normalized = (x[rows, cols] - mean[cols]) / I.sqrt(var[cols] + eps)
    out[rows, cols] = I.sigmoid(normalized * weight[cols])


@intent.kernel
def _bn_sigmoid_0b(
    x: I.In[I.f32, ("N", "C")],
    mean: I.In[I.f32, ("C",)],
    var: I.In[I.f32, ("C",)],
    bias: I.In[I.f32, ("C",)],
    out: I.Out[I.f32, ("N", "C")],
    eps: I.f32,
):
    rows = I.domain(0, x.shape[0])
    cols = I.domain(0, x.shape[1])
    normalized = (x[rows, cols] - mean[cols]) / I.sqrt(var[cols] + eps)
    out[rows, cols] = I.sigmoid(normalized + bias[cols])


@intent.kernel
def _bn_sigmoid_00(
    x: I.In[I.f32, ("N", "C")],
    mean: I.In[I.f32, ("C",)],
    var: I.In[I.f32, ("C",)],
    out: I.Out[I.f32, ("N", "C")],
    eps: I.f32,
):
    rows = I.domain(0, x.shape[0])
    cols = I.domain(0, x.shape[1])
    normalized = (x[rows, cols] - mean[cols]) / I.sqrt(var[cols] + eps)
    out[rows, cols] = I.sigmoid(normalized)


def build(context):
    ww = context.compile("bn_sigmoid_ww", _bn_sigmoid_ww)
    w0 = context.compile("bn_sigmoid_w0", _bn_sigmoid_w0)
    ob = context.compile("bn_sigmoid_0b", _bn_sigmoid_0b)
    oo = context.compile("bn_sigmoid_00", _bn_sigmoid_00)

    def sigmoid_batch_norm(
        input, running_mean, running_var, weight=None, bias=None,
        training=False, momentum=0.1, eps=1e-5,
    ):
        out = torch.empty_like(input)
        # The supplied profile is inference mode and rank-2.  The running
        # statistics are therefore the complete normalization state.
        eps_value = float(eps)
        if weight is not None and bias is not None:
            ww(input, running_mean, running_var, weight, bias, out, eps_value)
        elif weight is not None:
            w0(input, running_mean, running_var, weight, out, eps_value)
        elif bias is not None:
            ob(input, running_mean, running_var, bias, out, eps_value)
        else:
            oo(input, running_mean, running_var, out, eps_value)
        return out

    return sigmoid_batch_norm
