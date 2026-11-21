"""Original catalog input distributions shared by the native and JIT sides."""
from dataclasses import dataclass

@dataclass
class Arguments:
    arguments: tuple


def fused_add():
    import torch
    x = torch.randn((8192,4096),device="cuda",dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn((4096,),device=x.device,dtype=x.dtype)
    return Arguments((x,residual,weight,1/4096,1e-6,0.0))


def dropout_residual():
    import torch
    x = torch.randn((4096,4096),dtype=torch.bfloat16)*0.5
    residual = torch.randn_like(x)*0.5
    weight = torch.randn((4096,),dtype=x.dtype)
    dy = torch.randn_like(x)*0.25
    carry = torch.randn_like(x)*0.25
    return Arguments((x,residual,weight,dy,carry,17,0.9,1/0.9,1/4096,1e-6,1.0))
