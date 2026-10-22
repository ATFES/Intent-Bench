from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.activation.pointwise import relu_forward
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x = torch.randn((8192, 4096), device='cuda', dtype=torch.float16)
    (_, generated) = compile_single(context, relu_forward, (x,))
    return checked_call(generated, Tolerance(atol=0.0), cuda_graph=True, note='同算法；NaN 传播方式不同', owners=locals())
