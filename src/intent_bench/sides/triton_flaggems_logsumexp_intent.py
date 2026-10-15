from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.normalization.logsumexp import row_logsumexp
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    x = torch.randn((8192, 8192), device='cuda', dtype=torch.float32)
    (_, generated) = compile_single(context, row_logsumexp, (x,))
    return Side(generated, Tolerance(atol=0.0001, rtol=1e-05), cuda_graph=True, owners=locals())
