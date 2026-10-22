from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import cuda.tile as ct
import torch
from intent.examples.kernels.contraction.gemm import bf16_gemm
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (m, k, n) = (8192, 4096, 11008)
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
    with ct.compiler_timeout(context.compiler_timeout_seconds):
        (artifact, generated) = compile_single(context, bf16_gemm, (a, b))
    return checked_call(generated, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, owners=locals())
