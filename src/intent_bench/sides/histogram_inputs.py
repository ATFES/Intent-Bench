"""The original floating-key catalog case, captured once for all consumers."""
from .norm_inputs import Arguments

def samples():
    import torch
    values=torch.randint(0,256,(8*1024*1024,),device="cuda",dtype=torch.int32)
    return Arguments((values.to(torch.float32),))
