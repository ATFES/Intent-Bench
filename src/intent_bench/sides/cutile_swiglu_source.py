from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    shape = (8192, 14336)
    gate = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    up = torch.randn_like(gate)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/activation/swiglu/swiglu.py', 'swiglu', needs_utils=True)
    source = functional_launch(lambda : source_module.swiglu(gate, up))
    return checked_call(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
