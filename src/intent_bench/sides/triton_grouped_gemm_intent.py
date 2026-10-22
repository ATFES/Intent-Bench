from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.ragged.grouped_gemm import ragged_grouped_gemm
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (experts, rows, hidden, intermediate) = (8, 1024, 4096, 14336)
    x = torch.randn((experts * rows, hidden), device='cuda', dtype=torch.float16)
    weight = torch.randn((experts, hidden, intermediate), device='cuda', dtype=torch.float16)
    offsets = torch.arange(0, (experts + 1) * rows, rows, device='cuda', dtype=torch.int32)
    (_, generated) = compile_single(context, ragged_grouped_gemm, (x, offsets, weight))
    return checked_call(generated, Tolerance(atol=0.02, rtol=0.01), cuda_graph=False, owners=locals())
