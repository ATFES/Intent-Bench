from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import official_source

def prepare(context):
    (m, k, n) = (4096, 4096, 14336)
    a = torch.randn((m, k), device='cuda', dtype=torch.float16)
    b = torch.randn((k, n), device='cuda', dtype=torch.float16)
    source_module = official_source(context, 'corpus/gpu/baselines/cutile/cutile-python/gemm/dense/MatMul_runtime.py', 'intent_v2_cutile_dense_gemm')
    source = functional_launch(lambda : source_module.cutile_matmul(a, b, compiler_timeout=context.compiler_timeout_seconds))
    return Side(source, Tolerance(atol=0.01, rtol=0.01), cuda_graph=True, owners=locals())
