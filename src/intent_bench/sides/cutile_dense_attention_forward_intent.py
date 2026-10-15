from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.streaming.attention import flash_attention_bf16_fwd
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, query_heads, key_heads, sequence, dimension) = (2, 32, 8, 4096, 128)
    q = torch.randn((batch, query_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = dimension ** (-0.5)
    (_, generated) = compile_single(context, flash_attention_bf16_fwd, (q, k, v, scale), constexprs={'HEAD_GROUP': query_heads // key_heads, 'CAUSAL': True})
    return Side(generated, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
