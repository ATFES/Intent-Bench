import torch
import intent
import intent.language as I


@intent.kernel
def _tt(
    x: I.In[I.f32, ("N",)],
    mul: I.In[I.f32, ("N",)],
    sub: I.In[I.f32, ("N",)],
    alpha: I.f32,
    out: I.Out[I.f32, ("N",)],
):
    for i in I.parallel(I.domain(0, x.shape[0])):
        out[i] = x[i] * mul[i] - alpha * sub[i]


@intent.kernel
def _st(
    x: I.In[I.f32, ("N",)],
    mul: I.f32,
    sub: I.In[I.f32, ("N",)],
    alpha: I.f32,
    out: I.Out[I.f32, ("N",)],
):
    for i in I.parallel(I.domain(0, x.shape[0])):
        out[i] = x[i] * mul - alpha * sub[i]


@intent.kernel
def _ts(
    x: I.In[I.f32, ("N",)],
    mul: I.In[I.f32, ("N",)],
    sub: I.f32,
    alpha: I.f32,
    out: I.Out[I.f32, ("N",)],
):
    for i in I.parallel(I.domain(0, x.shape[0])):
        out[i] = x[i] * mul[i] - alpha * sub


@intent.kernel
def _ss(
    x: I.In[I.f32, ("N",)],
    mul: I.f32,
    sub: I.f32,
    alpha: I.f32,
    out: I.Out[I.f32, ("N",)],
):
    for i in I.parallel(I.domain(0, x.shape[0])):
        out[i] = x[i] * mul - alpha * sub


def build(context):
    tt = context.compile("mul_sub_tt", _tt)
    st = context.compile("mul_sub_st", _st)
    ts = context.compile("mul_sub_ts", _ts)
    ss = context.compile("mul_sub_ss", _ss)

    def mul_sub(input, other_mul, other_sub, alpha=1, out=None):
        if out is None:
            out = torch.empty_like(input)
        mul_tensor = isinstance(other_mul, torch.Tensor)
        sub_tensor = isinstance(other_sub, torch.Tensor)
        if mul_tensor and sub_tensor:
            tt(input, other_mul, other_sub, alpha, out)
        elif not mul_tensor and sub_tensor:
            st(input, other_mul, other_sub, alpha, out)
        elif mul_tensor and not sub_tensor:
            ts(input, other_mul, other_sub, alpha, out)
        else:
            ss(input, other_mul, other_sub, alpha, out)
        return out

    return mul_sub
