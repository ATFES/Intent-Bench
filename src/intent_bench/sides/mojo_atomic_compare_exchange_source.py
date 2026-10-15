from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    slots = 65536
    initial = torch.where(torch.arange(slots, dtype=torch.int32) % 3 == 0, torch.zeros((), dtype=torch.int32), torch.full((), 2, dtype=torch.int32))
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_claim_zero_slots')
    source_state = initial.clone()
    source_outputs = {'previous': None}

    def restore_source():
        source_state.copy_(initial)

    def source_launch():
        (_, source_outputs['previous']) = runtime.claim_zero_slots(source_state)
    source = PreparedLaunch(source_launch, lambda : (source_state, source_outputs['previous']), prepare=restore_source)
    return Side(source, (Tolerance(atol=0.0), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有 claim_zero_slots i32 N65536 InOut/Out；每次调用前恢复 state，恢复不计时；双方计完整 host 调用。', owners=locals())
