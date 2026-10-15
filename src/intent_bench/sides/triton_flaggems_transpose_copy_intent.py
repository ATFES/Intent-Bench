from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.layout.transpose import matrix_transpose
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    x = torch.randn((4093, 8191), device='cuda', dtype=torch.float16)
    (_, generated) = compile_single(context, matrix_transpose, (x,))
    return Side(generated, Tolerance(atol=0.0), cuda_graph=True, owners=locals())
