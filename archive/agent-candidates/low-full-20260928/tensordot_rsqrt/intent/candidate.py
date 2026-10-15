import torch
import intent
import intent.language as I


@intent.kernel
def contract_kernel(
    a: I.In[I.f32, ("M", "K")],
    b: I.In[I.f32, ("K", "N")],
    out: I.Out[I.f32, ("M", "N")],
):
    product = I.matmul(a[:, :], b[:, :], acc_dtype=I.f32)
    out[:, :] = product


@intent.kernel
def rsqrt_kernel(
    x: I.In[I.f32, ("M", "N")],
    out: I.Out[I.f32, ("M", "N")],
):
    out[:, :] = I.rsqrt(x[:, :])


def build(context):
    contract = context.compile("tensordot_contract", contract_kernel)
    rsqrt = context.compile("tensordot_rsqrt", rsqrt_kernel)

    def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims):
        intermediate = torch.empty((a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device)
        output = torch.empty_like(intermediate)
        contract(a, b, intermediate)
        rsqrt(intermediate, output)
        return output

    return tensordot_rsqrt
