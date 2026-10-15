from __future__ import annotations
from intent_bench.sides import Side
import math
import torch
from kernels.streaming.attention_specialized import gemma_gqa_prefill
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, sequence, query_heads, key_heads, dimension) = (2, 4096, 32, 8, 128)
    q = torch.randn((batch, query_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(dimension)
    (_, generated) = compile_single(context, gemma_gqa_prefill, (q, k, v, scale), constexprs={'HEAD_GROUP': query_heads // key_heads, 'WINDOW': 1024, 'SOFT_CAP': 50.0})
    return Side(generated, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, note='同算法；中间精度和舍入不同', owners=locals())
