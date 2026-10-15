from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    packed = torch.randn((4096, 28672), device='cuda', dtype=torch.bfloat16)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/activation/silu_and_mul/silu_and_mul.py', 'silu_and_mul')
    source = functional_launch(lambda : source_module.silu_and_mul(packed))
    return Side(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
