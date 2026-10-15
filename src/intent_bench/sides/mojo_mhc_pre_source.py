from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    factors = 1 + torch.arange(4).mul(0.01).view(1, -1, 1)
    residual = (torch.randn((2048, 4, 4096), dtype=torch.float32) * factors).bfloat16()
    weight = (torch.randn((24, 4, 4096), dtype=torch.float32) * 0.0001 * factors).flatten(1, 2)
    scale = torch.randn((3,), dtype=torch.float32) * 0.1
    base = torch.randn((24,), dtype=torch.float32) * 0.1
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')
    (generated, source) = ({}, {})

    def reference():
        source['output'] = runtime.mhc_pre(residual, weight, scale, base)
    return Side(PreparedLaunch(reference, lambda : source['output']), (Tolerance(atol=0.01, rtol=0.001), Tolerance(atol=0.01, rtol=0.001), Tolerance(atol=0.05, rtol=0.02)), cuda_graph=False, device_type='cpu', note='既有 T2048/H4096/streams4 MHC pre 两 kernel；weight 先转 bf16，f32 GEMM/RMS/混合，Sinkhorn10 次、原 epsilon 和容差；PyTorch CPU reference 沿用 source 的 mhc_pre_ref 算法，计完整 host 调用及分配，单 NUMA 8 核。', owners=locals())
