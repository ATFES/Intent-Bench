from __future__ import annotations
from intent_bench.sides import Side
import math
import torch
from kernels.streaming.attention import mla_prefill
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, heads, key_heads, sequence, dimension, rope) = (1, 128, 1, 2048, 128, 64)
    q = torch.randn((batch, heads, sequence, dimension), device='cuda', dtype=torch.float16)
    qpe = torch.randn((batch, heads, sequence, rope), device='cuda', dtype=torch.float16)
    k = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.float16)
    v = torch.randn_like(k)
    kpe = torch.randn((batch, 1, sequence, rope), device='cuda', dtype=torch.float16)
    scale = 1.0 / math.sqrt(dimension + rope)
    (_, generated) = compile_single(context, mla_prefill, (q, qpe, k, kpe, v, scale), constexprs={'HEAD_GROUP': heads // key_heads})
    return Side(generated, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
