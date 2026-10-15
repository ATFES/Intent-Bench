from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.activation.swiglu import silu_and_mul_packed
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    packed = torch.randn((4096, 28672), device='cuda', dtype=torch.bfloat16)
    (_, generated) = compile_single(context, silu_and_mul_packed, (packed,))
    return Side(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, note='同算法；source 使用 FTZ 与近似除法', owners=locals())
