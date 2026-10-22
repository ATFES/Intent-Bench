from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import torch
from intent.examples.kernels.streaming.attention import flash_gqa_attention_fwd
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, query_heads, key_heads, sequence, dimension) = (4, 32, 8, 4096, 128)
    q = torch.randn((batch, query_heads, sequence, dimension), device='cuda', dtype=torch.float16)
    k = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(dimension)
    (_, generated) = compile_single(context, flash_gqa_attention_fwd, (q, k, v, scale), constexprs={'HEAD_GROUP': query_heads // key_heads, 'CAUSAL': True})
    return checked_call(generated, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
