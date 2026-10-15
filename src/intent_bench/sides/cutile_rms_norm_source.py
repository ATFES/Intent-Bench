from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    hidden = 4096
    x = torch.randn((8192, hidden), device='cuda', dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device='cuda', dtype=torch.bfloat16)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/normalization/rms_norm/rms_norm.py', 'rms_norm', needs_utils=True)
    source = functional_launch(lambda : source_module.rms_norm(x, (hidden,), weight, 1e-05, mode='multi_wave_reload'))
    return Side(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, note='同算法；source 额外写 Rstd', owners=locals())
