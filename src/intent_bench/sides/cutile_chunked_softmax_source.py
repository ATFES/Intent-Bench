from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    x = torch.randn((8192, 32768), device='cuda', dtype=torch.bfloat16)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/normalization/softmax/softmax.py', 'chunked_softmax')
    source = functional_launch(lambda : source_module.softmax(x, use_tma=False, use_chunked=True))
    return checked_call(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())
