from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    x = torch.randn((8192, 4096), device='cuda', dtype=torch.float16)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/activation/fused/gelu.py', 'gelu')
    source = functional_launch(lambda : source_module.gelu(x, approximate='tanh'))
    return Side(source, Tolerance(atol=0.01, rtol=0.01), cuda_graph=True, note='同算法；source 保留 f16 中间舍入', owners=locals())
