from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import official_source

def prepare(context):
    (m, k, n, block) = (4096, 4096, 14336, 32)
    source_module = official_source(context, 'corpus/gpu/baselines/cutile/cutile-python/gemm/block_scaled/BlockScaledMatMul_runtime.py', 'intent_v2_cutile_block_scaled_gemm')
    (lhs, lhs_scale) = source_module.block_quantize(torch.randn((m, k), device='cuda', dtype=torch.float32), block)
    (rhs_rows, rhs_scale_rows) = source_module.block_quantize(torch.randn((n, k), device='cuda', dtype=torch.float32), block)
    rhs = rhs_rows.T
    rhs_scale = rhs_scale_rows.T
    source = functional_launch(lambda : source_module.cutile_block_scaled_matmul(lhs, lhs_scale, rhs, rhs_scale, compiler_timeout=context.compiler_timeout_seconds))
    return checked_call(source, Tolerance(atol=1.0, rtol=0.02), cuda_graph=True, owners=locals())
