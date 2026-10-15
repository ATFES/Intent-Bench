from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.activation.pointwise import addcmul_broadcast_bf16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    shape = (64, 128, 4096)
    x = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    scale = torch.randn(shape[:2], device='cuda', dtype=torch.bfloat16)
    bias = torch.randn(shape[:2], device='cuda', dtype=torch.bfloat16)
    (_, generated) = compile_single(context, addcmul_broadcast_bf16, (x, scale, bias))
    return Side(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())
