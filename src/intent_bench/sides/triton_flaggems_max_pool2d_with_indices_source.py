from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.references.torch_ops import max_pool2d_with_indices
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    x = torch.randn((8, 32, 128, 128), device='cuda', dtype=torch.float16)
    source = functional_launch(lambda : max_pool2d_with_indices(*(x,)))
    return Side(source, (Tolerance(atol=0.0), Tolerance(atol=0.0)), cuda_graph=True, owners=locals())
