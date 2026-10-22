from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.references.torch_ops import cumsum
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((8192, 8192), device='cuda', dtype=torch.float32)
    source = functional_launch(lambda : cumsum(*(x,)))
    return checked_call(source, Tolerance(atol=0.002, rtol=1e-05), cuda_graph=True, owners=locals())
