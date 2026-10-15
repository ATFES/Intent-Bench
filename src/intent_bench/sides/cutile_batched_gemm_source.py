from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (batch, m, k, n) = (32, 512, 1024, 512)
    a = torch.randn((batch, m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((batch, k, n), device='cuda', dtype=torch.bfloat16)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/gemm/batched/bmm.py', 'batched_gemm')
    source = functional_launch(lambda : source_module.bmm(a, b, transpose_a=False, transpose_b=False, static_persistent=True, compiler_timeout=context.compiler_timeout_seconds))
    return Side(source, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, owners=locals())
