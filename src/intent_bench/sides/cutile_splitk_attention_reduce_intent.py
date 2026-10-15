from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.streaming.splitk_reduce import splitk_attention_reduce
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, heads, splits, dimension) = (8, 32, 16, 128)
    partial = torch.randn((batch, heads, splits, dimension), device='cuda', dtype=torch.bfloat16)
    partial_lse = torch.randn((batch, heads, splits), device='cuda', dtype=torch.float32)
    (_, generated) = compile_single(context, splitk_attention_reduce, (partial, partial_lse))
    return Side(generated, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, owners=locals())
