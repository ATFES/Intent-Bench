from __future__ import annotations
from intent_bench.sides import Side
import math
import torch
from kernels.streaming.attention_specialized import attention_sink_prefill
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, sequence, key_heads, repeats, dimension) = (1, 4096, 8, 4, 128)
    query_heads = key_heads * repeats
    q = torch.randn((batch, query_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = torch.randn((query_heads,), device='cuda', dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(dimension)
    (_, generated) = compile_single(context, attention_sink_prefill, (q, k, v, sinks, scale), constexprs={'HEAD_GROUP': repeats})
    return Side(generated, Tolerance(atol=0.05, rtol=0.02), cuda_graph=False, note='同算法；source 的布局转换与辅助工作计入其调用时间', owners=locals())
