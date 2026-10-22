from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    x = torch.randn((4096, 28672), device='cuda', dtype=torch.float16)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/activation/fused/geglu.py', 'geglu', needs_gelu=True)
    source = functional_launch(lambda : source_module.geglu(x, dim=-1, approximate='tanh'))
    return checked_call(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, note='同算法；source 保留 f16 中间舍入', owners=locals())
