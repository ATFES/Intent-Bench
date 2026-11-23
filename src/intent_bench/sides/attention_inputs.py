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

def backward_source(root):
    from ..corpus._common.loading import load_module
    runtime=load_module(root/"corpus/gpu/baselines/triton/meta-applied-ai/support/runtime.py",
                        "intent_v2_triton_flash_backward_runtime")
    return runtime.load_source(root/"corpus/gpu/baselines/triton/meta-applied-ai/attention/flash_backward/flash_backward.py",
                               "intent_v2_triton_flash_backward_source")

def backward():
    import torch
    from ..corpus import PROJECT_ROOT
    from .triton_flash_attention_backward_source import _prepare_source_launch
    q=torch.randn((2,16,2048,128),device="cuda",dtype=torch.float16)*0.5
    k,v=torch.randn_like(q)*0.5,torch.randn_like(q)*0.5
    output=torch.empty_like(q)
    logs=torch.empty(q.shape[:-1],device=q.device,dtype=torch.float32)
    source=backward_source(PROJECT_ROOT)
    _prepare_source_launch(lambda:source.flash(q,k,v,output,logs))
    return Arguments((q,k,v,output,torch.randn_like(q)*0.05,logs,128**-0.5))
