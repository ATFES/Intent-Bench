"""Exact shared buffers for the three mixed-compute catalog workloads."""
import math
from .norm_inputs import Arguments

def natural_lse(q,k,v,out,grad,log2_lse,scale):
    return q,k,v,out,grad,log2_lse / math.log2(math.e),scale

def grouped():
    import torch
    q=torch.randn((8,32,1,128),device="cuda",dtype=torch.bfloat16)
    k=torch.randn((8,8,8192,128),device="cuda",dtype=torch.bfloat16)
    return Arguments((q,k,torch.randn_like(k),128**-0.5))

def latent():
    import torch
    def half(shape):
        return torch.randn(shape,dtype=torch.float16)
    q, qr = half((8,128,512)), half((8,128,64))
    k, kr = half((4096,16,1,512)), half((4096,16,1,64))
    offsets=torch.arange(0,4097,512,dtype=torch.int32)
    pages=torch.arange(4096,dtype=torch.int32)
    lengths=torch.full((8,),8192,dtype=torch.int32)
    return Arguments((q,qr,k,kr,offsets,pages,lengths,576**-0.5))

def backward():
    import torch
    from ..corpus.cpu.baselines.pytorch.cpu_runtime import causal_attention_with_log2_lse
    q=torch.randn((2,16,2048,128),device="cuda",dtype=torch.float16)*0.5
    k,v=torch.randn_like(q)*0.5,torch.randn_like(q)*0.5
    scale = q.shape[-1] ** -0.5
    output, logs = causal_attention_with_log2_lse(q.float(), k.float(), v.float(), scale)
    return Arguments((q,k,v,output.to(q.dtype),torch.randn_like(q)*0.05,logs,scale))
