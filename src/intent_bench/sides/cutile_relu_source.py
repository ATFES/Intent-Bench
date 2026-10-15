from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    x = torch.randn((8192, 4096), device='cuda', dtype=torch.float16)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/activation/relu/relu.py', 'relu')
    source = functional_launch(lambda : source_module.relu(x))
    return Side(source, Tolerance(atol=0.0), cuda_graph=True, note='同算法；NaN 传播方式不同', owners=locals())
