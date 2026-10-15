import torch
import intent
import intent.language as I


@intent.kernel
def _logsoftmax_linear_bias_kernel(
    x: I.In[I.f32, (32, 256)],
    w: I.In[I.f32, (256, 256)],
    b: I.In[I.f32, (256,)],
    y: I.Out[I.f32, (32, 256)],
):
    # linear: input @ weight.T  (weight is (out_features, in_features))
    logits = I.matmul(x, w, acc_dtype=I.f32, transpose_rhs=True)
    # add bias, broadcasts along the trailing (feature) axis
    logits = logits + b
    # numerically stable log-softmax along the last axis (dim == -1)
    mx = I.reduce.max(logits, axis=1)
    shifted = logits - I.reshape(mx, (32, 1))
    e = I.exp(shifted)
    s = I.reduce.sum(e, axis=1, acc_dtype=I.f32)
    lse = I.reshape(I.log(s), (32, 1))
    y[:, :] = shifted - lse


@intent.kernel
def _logsoftmax_linear_nobias_kernel(
    x: I.In[I.f32, (32, 256)],
    w: I.In[I.f32, (256, 256)],
    y: I.Out[I.f32, (32, 256)],
):
    logits = I.matmul(x, w, acc_dtype=I.f32, transpose_rhs=True)
    mx = I.reduce.max(logits, axis=1)
    shifted = logits - I.reshape(mx, (32, 1))
    e = I.exp(shifted)
    s = I.reduce.sum(e, axis=1, acc_dtype=I.f32)
    lse = I.reshape(I.log(s), (32, 1))
    y[:, :] = shifted - lse


def build(context):
    k_bias = context.compile("logsoftmax_linear_bias", _logsoftmax_linear_bias_kernel)
    k_nobias = context.compile(
        "logsoftmax_linear_nobias", _logsoftmax_linear_nobias_kernel
    )

    def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
        if dim not in (-1, 1):
            raise ValueError("this specialization supports dim=-1 only")
        if dtype is not None and dtype is not torch.float32:
            raise ValueError("this specialization supports dtype=None only")
        out = torch.empty_like(input)
        if bias is None:
            k_nobias(input, weight, out)
        else:
            k_bias(input, weight, bias, out)
        return out

    return log_softmax_linear

