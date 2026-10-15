import torch
import intent
import intent.language as I


@intent.kernel
def _fused_bias(
    x: I.In[I.f32, ("M", "K")],
    w: I.In[I.f32, ("N", "K")],
    b: I.In[I.f32, ("N",)],
    out: I.Out[I.f32, ("M", "N")],
):
    z = I.matmul(x, w, acc_dtype=I.f32, transpose_rhs=True)
    z = z + b
    mx = I.reduce.max(z, axis=1, acc_dtype=I.f32)
    mx2 = I.reshape(mx, (z.shape[0], 1))
    e = I.exp(z - mx2)
    s = I.reduce.sum(e, axis=1, acc_dtype=I.f32)
    lse = mx + I.log(s)
    out[:, :] = z - I.reshape(lse, (z.shape[0], 1))


@intent.kernel
def _fused_nobias(
    x: I.In[I.f32, ("M", "K")],
    w: I.In[I.f32, ("N", "K")],
    out: I.Out[I.f32, ("M", "N")],
):
    z = I.matmul(x, w, acc_dtype=I.f32, transpose_rhs=True)
    mx = I.reduce.max(z, axis=1, acc_dtype=I.f32)
    mx2 = I.reshape(mx, (z.shape[0], 1))
    e = I.exp(z - mx2)
    s = I.reduce.sum(e, axis=1, acc_dtype=I.f32)
    lse = mx + I.log(s)
    out[:, :] = z - I.reshape(lse, (z.shape[0], 1))


def build(context):
    fused_bias = context.compile("log_softmax_linear_bias", _fused_bias)
    fused_nobias = context.compile("log_softmax_linear_nobias", _fused_nobias)

    def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
        if dtype is not None:
            input = input.to(dtype)
        output = torch.empty(
            (*input.shape[:-1], weight.shape[0]),
            device=input.device,
            dtype=input.dtype,
        )
        x2 = input.reshape(-1, input.shape[-1])
        out2 = output.reshape(-1, weight.shape[0])
        if bias is None:
            fused_nobias(x2, weight, out2)
        else:
            fused_bias(x2, weight, bias, out2)
        return output

    return log_softmax_linear
