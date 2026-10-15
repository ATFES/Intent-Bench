from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.loss.cross_entropy import fused_cross_entropy
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
from intent_bench.observed import with_writes

def prepare(context):
    initial = torch.randn((4096, 16384), dtype=torch.float32) * 0.5
    labels = torch.randint(0, 16384, (4096,), dtype=torch.int32)
    labels[::17] = -100
    artifact = intent.compile(fused_cross_entropy, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options, constexprs={'IGNORE_INDEX': -100})

    def side(function):
        logits = initial.clone()
        state = {}

        def launch():
            state['outputs'] = function(logits, labels)
        return PreparedLaunch(launch, lambda : state['outputs'], prepare=lambda : logits.copy_(initial))
    return Side(side(with_writes(artifact)), (Tolerance(atol=2e-05), Tolerance(atol=0.0002), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有4096x16384 f32 fused cross entropy，含原地logits gradient、loss和prediction；原ignore_index -100；单NUMA8核，PyTorch CPU reference，完整host调用；每次恢复logits且恢复不计时。', owners=locals())
