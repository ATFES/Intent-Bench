from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (batch, sequence, key_heads, repeats, dimension) = (1, 4096, 8, 4, 128)
    query_heads = key_heads * repeats
    q = torch.randn((batch, query_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = torch.randn((query_heads,), device='cuda', dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(dimension)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/attention/sink_prefill/attention_sink.py', 'attention_sink_prefill')
    q_source = q.permute(0, 2, 1, 3).contiguous().view(batch, sequence, key_heads, repeats, dimension)
    k_source = k.permute(0, 2, 1, 3).contiguous()
    v_source = v.permute(0, 2, 1, 3).contiguous()
    start = torch.zeros((1,), device='cuda', dtype=torch.int32)

    def source_call():
        output = source_module.attention_sink(q_source, k_source, v_source, sinks, scale, None, start)
        return output.view(batch, sequence, query_heads, dimension).permute(0, 2, 1, 3)
    source = functional_launch(source_call)
    return checked_call(source, Tolerance(atol=0.05, rtol=0.02), cuda_graph=False, note='同算法；source 的布局转换与辅助工作计入其调用时间', owners=locals())
