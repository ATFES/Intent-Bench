from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.references.torch_ops import embedding
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, sequence, vocabulary, hidden) = (8, 2048, 32768, 4096)
    table = torch.randn((vocabulary, hidden), device='cuda', dtype=torch.bfloat16)
    indices = torch.randint(0, vocabulary, (batch, sequence), device='cuda', dtype=torch.int64)
    source = functional_launch(lambda : embedding(*(indices, table)))
    return checked_call(source, Tolerance(atol=0.0), cuda_graph=True, owners=locals())
