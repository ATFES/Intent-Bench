from __future__ import annotations
from intent_bench.sides import Side
import torch
import triton
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (m, n, k, splits, block) = (4096, 14336, 4096, 4, 256)
    lhs_storage = torch.randn((m, k), device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    rhs_storage = torch.randn((k, n), device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    block_m = block_n = 64
    generated_output = torch.zeros((m, n), device='cuda', dtype=torch.float16)
    source_module = load_module(context.project_root / 'corpus/gpu/baselines/triton/meta-applied-ai/gemm/fp8_scaled/scaled_fp8_gemm.py', 'intent_v2_triton_scaled_fp8_splitk')
    source_output = torch.zeros_like(generated_output)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n), splits)

    def launch():
        source_module.scaled_gemm_splitk[grid](lhs_storage, rhs_storage, source_output, lhs_storage.stride(0), lhs_storage.stride(1), rhs_storage.stride(0), rhs_storage.stride(1), source_output.stride(0), source_output.stride(1), 1.0, 1.0, m, n, k, block_m, block_n, block, splits, 8, num_stages=3, num_warps=8)
    launch()
    source = PreparedLaunch(launch=launch, outputs=lambda : source_output, prepare=lambda : source_output.zero_())
    return Side(source, Tolerance(atol=0.5, rtol=0.01), cuda_graph=False, owners=locals())
