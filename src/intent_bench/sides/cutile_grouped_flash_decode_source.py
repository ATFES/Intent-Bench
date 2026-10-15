from __future__ import annotations
from intent_bench.sides import Side
import math
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (batch, query_heads, key_heads, sequence, dimension, split_size) = (8, 32, 8, 8192, 128, 256)
    q = torch.randn((batch, query_heads, 1, dimension), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(dimension)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/attention/flash_decode/flash_decode.py', 'grouped_flash_decode', needs_utils=True, needs_splitk=True)
    source = functional_launch(lambda : source_module.fmha_decode(q, k, v, scale, kv_len_per_split=split_size))
    return Side(source, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
