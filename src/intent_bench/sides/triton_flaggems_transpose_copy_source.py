from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.references.torch_ops import transpose_copy
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((4093, 8191), device='cuda', dtype=torch.float16)
    source = functional_launch(lambda : transpose_copy(*(x,)))
    return checked_call(source, Tolerance(atol=0.0), cuda_graph=True, owners=locals())
