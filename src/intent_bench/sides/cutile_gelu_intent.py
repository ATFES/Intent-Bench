from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.activation.pointwise import gelu_tanh
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((8192, 4096), device='cuda', dtype=torch.float16)
    (_, generated) = compile_single(context, gelu_tanh, (x,))
    return checked_call(generated, Tolerance(atol=0.01, rtol=0.01), cuda_graph=True, note='同算法；source 保留 f16 中间舍入', owners=locals())
