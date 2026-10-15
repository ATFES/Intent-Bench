from __future__ import annotations
from intent_bench.sides import Side
import cuda.tile as ct
import torch
from kernels.contraction.gemm import Activation
from kernels.contraction.gemm import gemm
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (m, k, n) = (4096, 4096, 14336)
    a = torch.randn((m, k), device='cuda', dtype=torch.float16)
    b = torch.randn((k, n), device='cuda', dtype=torch.float16)
    with ct.compiler_timeout(context.compiler_timeout_seconds):
        (artifact, generated) = compile_single(context, gemm, (a, b), constexprs={'ACTIVATION': Activation.NONE})
    return Side(generated, Tolerance(atol=0.01, rtol=0.01), cuda_graph=True, owners=locals())
