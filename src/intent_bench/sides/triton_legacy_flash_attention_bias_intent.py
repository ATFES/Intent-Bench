from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.streaming.attention import flash_attention_bias_fwd
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, heads, sequence, dimension) = (4, 32, 4096, 128)
    shape = (batch, heads, sequence, dimension)
    q = torch.randn(shape, device='cuda', dtype=torch.float16) * 0.5
    k = torch.randn_like(q) * 0.5
    v = torch.randn_like(q) * 0.5
    bias = torch.randn((batch, heads, sequence), device='cuda', dtype=torch.float32) * 0.125
    scale = dimension ** (-0.5)
    (_, generated) = compile_single(context, flash_attention_bias_fwd, (q, k, v, bias, scale))
    return Side(generated, Tolerance(atol=0.02, rtol=0.02), cuda_graph=True, note='同算法；source 额外写 LSE 和临时数据。', owners=locals())
