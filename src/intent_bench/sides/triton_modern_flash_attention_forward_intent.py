from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.streaming.attention import flash_attention_fwd
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, heads, sequence, dimension) = (2, 16, 2048, 128)
    shape = (batch, heads, sequence, dimension)
    q = torch.randn(shape, device='cuda', dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = dimension ** (-0.5)
    (_, generated) = compile_single(context, flash_attention_fwd, (q, k, v, scale), constexprs={'CAUSAL': True})
    return checked_call(generated, Tolerance(atol=0.02, rtol=0.02), cuda_graph=True, note='source launch uses num_stages=1 and num_warps=4.', owners=locals())
