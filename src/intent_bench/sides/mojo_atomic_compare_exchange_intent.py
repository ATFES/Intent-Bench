from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.synchronization.compare_exchange import claim_zero_slots
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import intent_call
from intent_bench.cpu_support import report_selection

def prepare(context):
    slots = 65536
    initial = torch.where(torch.arange(slots, dtype=torch.int32) % 3 == 0, torch.zeros((), dtype=torch.int32), torch.full((), 2, dtype=torch.int32))
    artifact = context.compile(claim_zero_slots)
    generated_state = initial.clone()

    def restore_generated():
        generated_state.copy_(initial)

    generated = intent_call(artifact, (generated_state,), reset=restore_generated,
                            observe=lambda output: selected_outputs(artifact, (generated_state, output)))
    return checked_call(generated, (Tolerance(atol=0.0), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有 claim_zero_slots i32 N65536 InOut/Out；每次调用前恢复 state，恢复不计时；默认计准备后的 native 执行；Python 调用仅在 host_call 模式计时。', owners=locals())

def selected_outputs(artifact, outputs):
    report_selection(artifact.observation)
    return outputs
