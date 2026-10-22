from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import torch
from intent.examples.kernels.streaming.mla import absorbed_mla_decode
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, heads, sequence, latent, rope) = (8, 64, 8192, 512, 64)
    q = torch.randn((batch, heads, latent), device='cuda', dtype=torch.float16)
    qpe = torch.randn((batch, heads, rope), device='cuda', dtype=torch.float16)
    cache = torch.randn((batch, sequence, latent), device='cuda', dtype=torch.float16)
    cache_pe = torch.randn((batch, sequence, rope), device='cuda', dtype=torch.float16)
    scale = 1.0 / math.sqrt(latent + rope)
    (_, generated) = compile_single(context, absorbed_mla_decode, (q, qpe, cache, cache_pe, scale))
    return checked_call(generated, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, note='同算法；source 使用近似除法并额外输出 LSE', owners=locals())
