from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.references.torch_ops import addcmul
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    shape = (64, 128, 4096)
    x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    scale = torch.randn(shape[:2], device='cuda', dtype=torch.bfloat16)
    bias = torch.randn(shape[:2], device='cuda', dtype=torch.bfloat16)
    source = functional_launch(lambda : addcmul(*(x, scale, bias)))
    return Side(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())
