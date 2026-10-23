from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import intent
import torch
from intent.examples.kernels.factorization.triangular_solve import batched_lower_triangular_solve
from intent_bench.implementations.context import Tolerance

def prepare(context):
    lower = torch.tril(torch.randn((4096, 16, 16), dtype=torch.float32))
    lower.diagonal(dim1=-2, dim2=-1).add_(2.0)
    initial = torch.randn((4096, 16), dtype=torch.float32)
    artifact = context.compile(batched_lower_triangular_solve)

    def side():
        solution = initial.clone()
        return intent_call(artifact, (lower, solution), reset=lambda: solution.copy_(initial), observe=lambda _: solution)
    return checked_call(side(), Tolerance(atol=0.0001, rtol=0.0001), cuda_graph=False, device_type='cpu', note='既有 B4096-N16 f32 lower-triangular solve；PyTorch eager CPU reference 保持逐列有序减法；单 NUMA 8 核；Intent默认计prepared native调用，每次恢复相同 RHS，恢复不计时。', owners=locals())
