from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import official_source

def prepare(context):
    hidden = 4096
    x = torch.randn((8192, hidden), device='cuda', dtype=torch.bfloat16)
    weight = torch.randn((hidden,), device='cuda', dtype=torch.bfloat16)
    bias = torch.randn((hidden,), device='cuda', dtype=torch.bfloat16)
    source_module = official_source(context, 'corpus/gpu/baselines/cutile/cutile-python/normalization/layer_norm/LayerNorm_runtime.py', 'intent_v2_cutile_layer_norm')
    source = functional_launch(lambda : source_module.cutile_layer_norm(x, weight, bias, 1e-05))
    return Side(source, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, note='同算法；source 额外写 Mean/Rstd', owners=locals())
