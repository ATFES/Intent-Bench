from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.activation.pointwise import gelu_tanh
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    x = torch.randn((8192, 4096), device='cuda', dtype=torch.float16)
    (_, generated) = compile_single(context, gelu_tanh, (x,))
    return Side(generated, Tolerance(atol=0.01, rtol=0.01), cuda_graph=True, note='同算法；source 保留 f16 中间舍入', owners=locals())
