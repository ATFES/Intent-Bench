from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import cuda.tile as ct
import torch
from intent.examples.kernels.contraction.batched_gemm import batched_gemm_nn
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, m, k, n) = (32, 512, 1024, 512)
    a = torch.randn((batch, m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((batch, k, n), device='cuda', dtype=torch.bfloat16)
    with ct.compiler_timeout(context.compiler_timeout_seconds):
        (artifact, generated) = compile_single(context, batched_gemm_nn, (a, b))
    return checked_call(generated, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, owners=locals())
