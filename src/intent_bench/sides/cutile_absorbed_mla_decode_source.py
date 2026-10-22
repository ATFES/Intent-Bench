from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (batch, heads, sequence, latent, rope) = (8, 64, 8192, 512, 64)
    q = torch.randn((batch, heads, latent), device='cuda', dtype=torch.float16)
    qpe = torch.randn((batch, heads, rope), device='cuda', dtype=torch.float16)
    cache = torch.randn((batch, sequence, latent), device='cuda', dtype=torch.float16)
    cache_pe = torch.randn((batch, sequence, rope), device='cuda', dtype=torch.float16)
    scale = 1.0 / math.sqrt(latent + rope)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/attention/mla_decode/mla_decoding.py', 'absorbed_mla_decode')
    source = functional_launch(lambda : source_module.mla_decoding(q, qpe, cache, cache_pe, scale)[0])
    return checked_call(source, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, note='同算法；source 使用近似除法并额外输出 LSE', owners=locals())
