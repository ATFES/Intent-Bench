from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.contraction.block_scaled import block_scaled_matmul
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance
from intent_bench.source_support import official_source

def prepare(context):
    (m, k, n, block) = (4096, 4096, 14336, 32)
    source_module = official_source(context, 'corpus/gpu/baselines/cutile/cutile-python/gemm/block_scaled/BlockScaledMatMul_runtime.py', 'intent_v2_cutile_block_scaled_gemm')
    (lhs, lhs_scale) = source_module.block_quantize(torch.randn((m, k), device='cuda', dtype=torch.float32), block)
    (rhs_rows, rhs_scale_rows) = source_module.block_quantize(torch.randn((n, k), device='cuda', dtype=torch.float32), block)
    rhs = rhs_rows.T
    rhs_scale = rhs_scale_rows.T
    arguments = (lhs.view(m, k // block, block), lhs_scale.view(torch.uint8), rhs.view(k // block, block, n), rhs_scale.view(torch.uint8))
    (artifact, generated) = compile_single(context, block_scaled_matmul, arguments)
    return Side(generated, Tolerance(atol=1.0, rtol=0.02), cuda_graph=True, owners=locals())
