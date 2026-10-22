from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (m, k, n) = (4096, 4096, 14336)
    a = torch.randn((m, k), device='cuda', dtype=torch.float16)
    b = torch.randn((k, n), device='cuda', dtype=torch.float16)
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/triton/gemm/dense/03-matrix-multiplication_runtime.py', 'intent_v2_triton_dense_gemm')
    source_function = runtime.load_matmul()
    source = functional_launch(lambda : source_function(a, b))
    return checked_call(source, Tolerance(atol=0.01, rtol=0.01), cuda_graph=True, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
