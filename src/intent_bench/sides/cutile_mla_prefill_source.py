from __future__ import annotations
from intent_bench.sides import Side
import math
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (batch, heads, key_heads, sequence, dimension, rope) = (1, 128, 1, 2048, 128, 64)
    q = torch.randn((batch, heads, sequence, dimension), device='cuda', dtype=torch.float16)
    qpe = torch.randn((batch, heads, sequence, rope), device='cuda', dtype=torch.float16)
    k = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    kpe = torch.randn((batch, 1, sequence, rope), device='cuda', dtype=torch.float16)
    scale = 1.0 / math.sqrt(dimension + rope)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/attention/mla/mla.py', 'mla_prefill')
    source = functional_launch(lambda : source_module.tile_mla(q, k, v, qpe, kpe, is_causal=True, scaling=scale))
    return Side(source, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
