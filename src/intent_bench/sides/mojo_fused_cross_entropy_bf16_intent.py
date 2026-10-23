from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import intent
import torch
from intent.examples.kernels.loss.cross_entropy import fused_cross_entropy_bf16
from intent_bench.implementations.context import Tolerance

def prepare(context):
    initial = torch.randn((8192, 32768), dtype=torch.bfloat16)
    labels = torch.randint(0, 32768, (8192,), dtype=torch.int64)
    labels[::17] = -100
    artifact = context.compile(fused_cross_entropy_bf16, constexprs={'IGNORE_INDEX': -100})

    def side():
        logits = initial.clone()
        return intent_call(artifact, (logits, labels), reset=lambda: logits.copy_(initial),
                           observe=lambda outputs: (logits, *outputs))
    return checked_call(side(), (Tolerance(atol=0.05, rtol=0.05), Tolerance(atol=0.05, rtol=0.01), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有 8192x32768 bf16 logits、i64 labels；f32 loss、i64 prediction、原地 bf16 gradient；IGNORE_INDEX=-100，沿用各输出原容差；PyTorch CPU reference，单 NUMA 8 核，prepared native调用；恢复 logits 不计时。', owners=locals())
