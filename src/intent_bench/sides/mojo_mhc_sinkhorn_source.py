from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    initial = torch.randn((8192, 4, 4), dtype=torch.float32)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

    def side(function):
        values = initial.clone()
        return PreparedLaunch(lambda : function(values), lambda : values, prepare=lambda : values.copy_(initial))
    return Side(side(runtime.mhc_sinkhorn), Tolerance(atol=0.0002, rtol=0.0001), cuda_graph=False, device_type='cpu', note='既有 T8192/streams4 f32 MHC Sinkhorn；exp 后固定 20 次行列归一化、无 epsilon；每次恢复独立输入且恢复不计时，PyTorch CPU reference、原容差，完整 host 调用，单 NUMA 8 核。', owners=locals())
