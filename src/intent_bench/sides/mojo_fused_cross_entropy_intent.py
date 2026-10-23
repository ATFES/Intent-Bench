from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import intent
import torch
from intent.examples.kernels.loss.cross_entropy import fused_cross_entropy
from intent_bench.implementations.context import Tolerance

def prepare(context):
    initial = torch.randn((4096, 16384), dtype=torch.float32) * 0.5
    labels = torch.randint(0, 16384, (4096,), dtype=torch.int32)
    labels[::17] = -100
    artifact = context.compile(fused_cross_entropy, constexprs={'IGNORE_INDEX': -100})

    def side():
        logits = initial.clone()
        return intent_call(artifact, (logits, labels), reset=lambda: logits.copy_(initial),
                           observe=lambda outputs: (logits, *outputs))
    return checked_call(side(), (Tolerance(atol=2e-05), Tolerance(atol=0.0002), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有4096x16384 f32 fused cross entropy，含原地logits gradient、loss和prediction；原ignore_index -100；单NUMA8核，PyTorch CPU reference，prepared native调用；每次恢复logits且恢复不计时。', owners=locals())
