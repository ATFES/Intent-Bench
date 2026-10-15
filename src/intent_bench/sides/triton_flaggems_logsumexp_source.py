from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.references.torch_ops import logsumexp
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    x = torch.randn((8192, 8192), device='cuda', dtype=torch.float32)
    source = functional_launch(lambda : logsumexp(*(x,)))
    return Side(source, Tolerance(atol=0.0001, rtol=1e-05), cuda_graph=True, owners=locals())
