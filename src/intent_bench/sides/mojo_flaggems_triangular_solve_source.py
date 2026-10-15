from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    lower = torch.tril(torch.randn((4096, 16, 16), dtype=torch.float32))
    lower.diagonal(dim1=-2, dim2=-1).add_(2.0)
    initial = torch.randn((4096, 16), dtype=torch.float32)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

    def side(function):
        solution = initial.clone()
        return PreparedLaunch(lambda : function(lower, solution), lambda : solution, prepare=lambda : solution.copy_(initial))
    return Side(side(runtime.triangular_solve), Tolerance(atol=0.0001, rtol=0.0001), cuda_graph=False, device_type='cpu', note='既有 B4096-N16 f32 lower-triangular solve；PyTorch eager CPU reference 保持逐列有序减法；单 NUMA 8 核；双方完整 host 调用，每次恢复相同 RHS，恢复不计时。', owners=locals())
