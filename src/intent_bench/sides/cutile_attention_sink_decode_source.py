from __future__ import annotations
from intent_bench.sides import Side
import math
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (batch, sequence, key_heads, head_group, dimension, split_size) = (32, 8192, 8, 4, 128, 256)
    query_heads = key_heads * head_group
    q_source = torch.randn((batch, 1, key_heads, head_group, dimension), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((batch, sequence, key_heads, dimension), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = torch.randn((query_heads,), device='cuda', dtype=torch.bfloat16)
    start_q = torch.tensor([sequence - 1], device='cuda', dtype=torch.int32)
    scale = 1.0 / math.sqrt(dimension)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/attention/sink_decode/attention_sink_decode.py', 'attention_sink_decode', needs_utils=True, needs_splitk=True)
    source = functional_launch(lambda : source_module.attention_sink_decode(q_source, k, v, sinks, scale, start_q=start_q, kv_len_per_split=split_size))
    return Side(source, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, note='同算法；source 使用 bf16 LSE 与近似除法', owners=locals())
