from __future__ import annotations
from intent_bench.sides import Side
import math
import torch
from kernels.streaming.attention_specialized import gemma_gqa_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_reduce
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, query_heads, key_heads, sequence, dimension, split_size) = (32, 32, 8, 8192, 128, 256)
    splits = (sequence + split_size - 1) // split_size
    q = torch.randn((batch, query_heads, 1, dimension), device='cuda', dtype=torch.bfloat16)
    k = torch.randn((batch, key_heads, sequence, dimension), device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(dimension)
    (_, partials) = compile_single(context, gemma_gqa_decode_partials, (q, k, v, scale), constexprs={'HEAD_GROUP': query_heads // key_heads, 'WINDOW': 1024, 'SOFT_CAP': 50.0, 'P': splits})
    (partial_lse, partial_output) = partials.outputs()
    (_, reduction) = compile_single(context, splitk_attention_reduce, (partial_output, partial_lse))

    def generated_launch():
        partials.launch()
        reduction.launch()
    generated = PreparedLaunch(generated_launch, lambda : reduction.outputs().unsqueeze(2))
    return Side(generated, Tolerance(atol=0.1, rtol=0.05), cuda_graph=False, note='同算法；中间精度和舍入不同', owners=locals())
