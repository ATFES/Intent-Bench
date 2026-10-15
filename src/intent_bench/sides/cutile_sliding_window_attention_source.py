from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (batch, sequence, query_heads, key_heads, dimension) = (2, 4096, 32, 8, 128)
    q = torch.randn((batch, query_heads, sequence, dimension), device='cuda', dtype=torch.float16)
    k = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/attention/sliding_window/swa_attention.py', 'sliding_window_attention')
    source = functional_launch(lambda : source_module.tile_swa_attention(q, k, v, window_size=1024, is_causal=True))
    return Side(source, Tolerance(atol=0.05, rtol=0.02), cuda_graph=False, note='同算法；source 的布局转换与辅助工作计入其调用时间', owners=locals())
