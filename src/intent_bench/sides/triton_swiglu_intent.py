from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.activation.swiglu import swiglu_forward
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    shape = (8192, 14336)
    gate = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    up = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    (_, generated) = compile_single(context, swiglu_forward, (gate, up))
    return Side(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=True, owners=locals())
