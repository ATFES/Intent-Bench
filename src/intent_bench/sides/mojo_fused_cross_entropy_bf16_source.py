from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    initial = torch.randn((8192, 32768), dtype=torch.bfloat16)
    labels = torch.randint(0, 32768, (8192,), dtype=torch.int64)
    labels[::17] = -100
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

    def side(function):
        logits = initial.clone()
        state = {}

        def launch():
            state['outputs'] = function(logits, labels)
        return PreparedLaunch(launch, lambda : state['outputs'], prepare=lambda : logits.copy_(initial))
    return Side(side(runtime.fused_cross_entropy_bf16), (Tolerance(atol=0.05, rtol=0.05), Tolerance(atol=0.05, rtol=0.01), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有 8192x32768 bf16 logits、i64 labels；f32 loss、i64 prediction、原地 bf16 gradient；IGNORE_INDEX=-100，沿用各输出原容差；PyTorch CPU reference，单 NUMA 8 核，完整 host 调用；恢复 logits 不计时。', owners=locals())
