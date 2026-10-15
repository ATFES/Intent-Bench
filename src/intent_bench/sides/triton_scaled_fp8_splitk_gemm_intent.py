from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.contraction.block_scaled import scaled_fp8_splitk_matmul
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (m, n, k, splits, block) = (4096, 14336, 4096, 4, 256)
    iterations = k // (splits * block)
    lhs_storage = torch.randn((m, k), device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    rhs_storage = torch.randn((k, n), device='cuda', dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    lhs = lhs_storage.view(m, iterations, splits, block)
    rhs = rhs_storage.view(iterations, splits, block, n)
    generated_output = torch.zeros((m, n), device='cuda', dtype=torch.float16)
    (_, generated_base) = compile_single(context, scaled_fp8_splitk_matmul, (lhs, rhs, generated_output, 1.0, 1.0))
    generated = PreparedLaunch(launch=generated_base.launch, outputs=lambda : generated_output, prepare=lambda : generated_output.zero_())
    return Side(generated, Tolerance(atol=0.5, rtol=0.01), cuda_graph=False, owners=locals())
