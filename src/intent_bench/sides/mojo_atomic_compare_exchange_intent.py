from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.synchronization.compare_exchange import claim_zero_slots
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
from intent_bench.cpu_support import report_selection

def prepare(context):
    slots = 65536
    initial = torch.where(torch.arange(slots, dtype=torch.int32) % 3 == 0, torch.zeros((), dtype=torch.int32), torch.full((), 2, dtype=torch.int32))
    artifact = intent.compile(claim_zero_slots, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    generated_state = initial.clone()
    generated_outputs = {'previous': None}

    def restore_generated():
        generated_state.copy_(initial)

    def generated_launch():
        generated_outputs['previous'] = artifact.run(generated_state)
    generated = PreparedLaunch(generated_launch, lambda : selected_outputs(artifact, (generated_state, generated_outputs['previous'])), prepare=restore_generated)
    return Side(generated, (Tolerance(atol=0.0), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有 claim_zero_slots i32 N65536 InOut/Out；每次调用前恢复 state，恢复不计时；双方计完整 host 调用。', owners=locals())

def selected_outputs(artifact, outputs):
    report_selection(artifact.observation)
    return outputs
