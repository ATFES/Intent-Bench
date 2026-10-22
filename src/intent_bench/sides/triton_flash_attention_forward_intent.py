from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import math
import torch
from intent.examples.kernels.streaming.attention import flash_attention_fwd
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, heads, sequence, dimension) = (4, 32, 4096, 128)
    shape = (batch, heads, sequence, dimension)
    q = torch.randn(shape, device='cuda', dtype=torch.float16)
    k = torch.randn(shape, device='cuda', dtype=torch.float16)
    v = torch.randn(shape, device='cuda', dtype=torch.float16)
    scale = 1.0 / math.sqrt(dimension)
    (_, generated) = compile_single(context, flash_attention_fwd, (q, k, v, scale), constexprs={'CAUSAL': True})
    return checked_call(generated, Tolerance(atol=0.02, rtol=0.02), cuda_graph=True, note='同算法；CUDA graph 设备执行计时；source 额外写 f32 M 辅助结果。', owners=locals())
