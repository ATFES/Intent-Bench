from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import tilegym_source

def prepare(context):
    (m, k, n) = (8192, 4096, 11008)
    a = torch.randn((m, k), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((k, n), device='cuda', dtype=torch.bfloat16)
    source_module = tilegym_source(context, 'corpus/gpu/baselines/cutile/tilegym/gemm/dense/matmul.py', 'tilegym_dense_gemm')
    source = functional_launch(lambda : source_module.matmul(a, b, trans_a=False, trans_b=False, static_persistent=True, compiler_timeout=context.compiler_timeout_seconds))
    return Side(source, Tolerance(atol=0.05, rtol=0.02), cuda_graph=True, owners=locals())
