from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    x = torch.randn((2048, 16384), dtype=torch.bfloat16)
    weight = torch.randn((16384, 24), dtype=torch.bfloat16)
    bias = torch.randn((24,), dtype=torch.bfloat16)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')
    (generated, source) = ({}, {})

    def reference():
        source['output'] = runtime.mhc_gemm_rms_scale(x, weight, bias)
    return Side(PreparedLaunch(reference, lambda : source['output']), (Tolerance(atol=0.05, rtol=0.02), Tolerance(atol=0.005, rtol=0.001)), cuda_graph=False, device_type='cpu', note='既有 T2048/H4096/streams4/P16 bf16 MHC GEMM+RMS；partial/finalize 两 kernel 完整 host 调用，含中间及输出分配；PyTorch CPU reference，保留 split16、bf16 mixed、f32 rms 与原容差，单 NUMA 8 核。', owners=locals())
