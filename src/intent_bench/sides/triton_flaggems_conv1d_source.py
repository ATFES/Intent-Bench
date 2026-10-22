from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.references.torch_ops import conv1d_same
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, length, width) = (64, 16384, 5)
    x = torch.randn((batch, length), device='cuda', dtype=torch.float16)
    weight = torch.randn((width,), device='cuda', dtype=torch.float16)
    source = functional_launch(lambda : conv1d_same(*(x, weight)))
    return checked_call(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())
