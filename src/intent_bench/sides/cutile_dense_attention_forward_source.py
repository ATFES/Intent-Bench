from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (batch, query_heads, key_heads, sequence, dimension) = (2, 32, 8, 4096, 128)
    q = torch.randn((batch, query_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/attention/dense/attention.py', 'dense_attention_forward')
    source = functional_launch(lambda : source_module.tile_fmha(q, k, v, is_causal=True))
    return Side(source, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
