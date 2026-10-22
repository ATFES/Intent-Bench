from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    left = torch.randn((3840, 4096), dtype=torch.float16)
    right = torch.randn_like(left)
    offsets = torch.tensor((0, 256, 768, 1792, 3840), dtype=torch.int32)
    generated = torch.empty((4, 4096, 4096), dtype=torch.float16)
    source = torch.empty_like(generated)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')
    return checked_call(PreparedCall(lambda : runtime.ragged_grouped_gemm_backward_weight(left, right, offsets, source), lambda : source), Tolerance(atol=0.05, rtol=0.02), cuda_graph=False, device_type='cpu', note='既有 rows256/512/1024/2048、K4096/N4096、f16 grouped GEMM weight backward；f32 累加，完整分组调用；双方输出预分配，PyTorch CPU reference，单 NUMA 8 核，沿用原容差。', owners=locals())
