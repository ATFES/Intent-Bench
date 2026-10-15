from __future__ import annotations
from intent_bench.sides import Side
import math
import torch
from kernels.streaming.attention_specialized import attention_sink_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_reduce
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, sequence, key_heads, head_group, dimension, split_size) = (32, 8192, 8, 4, 128, 256)
    query_heads = key_heads * head_group
    splits = (sequence + split_size - 1) // split_size
    q_source = torch.randn((batch, 1, key_heads, head_group, dimension), device='cuda', dtype=torch.bfloat16)
    q = q_source.view(batch, query_heads, dimension)
    k = torch.randn((batch, sequence, key_heads, dimension), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    sinks = torch.randn((query_heads,), device='cuda', dtype=torch.bfloat16)
    start_q = torch.tensor([sequence - 1], device='cuda', dtype=torch.int32)
    scale = 1.0 / math.sqrt(dimension)
    (_, partials) = compile_single(context, attention_sink_decode_partials, (q, k, v, sinks, start_q, scale), constexprs={'HEAD_GROUP': head_group, 'WINDOW': 0, 'P': splits})
    (partial_lse, partial_output) = partials.outputs()
    (_, reduction) = compile_single(context, splitk_attention_reduce, (partial_output, partial_lse))

    def generated_launch():
        partials.launch()
        reduction.launch()
    generated = PreparedLaunch(generated_launch, lambda : reduction.outputs().reshape(batch, 1, query_heads * dimension))
    return Side(generated, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, note='同算法；source 使用 bf16 LSE 与近似除法', owners=locals())
