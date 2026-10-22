from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.model import PreparedCall
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (batch, hidden, width) = (32, 4096, 4)
    x = torch.randn((batch, hidden), device='cuda', dtype=torch.bfloat16)
    initial = torch.randn((batch, hidden, width), device='cuda', dtype=torch.bfloat16)
    weight = torch.randn((hidden, width), device='cuda', dtype=torch.float32)
    bias = torch.randn((hidden,), device='cuda', dtype=torch.float32)
    runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/fla/conv/causal1d/causal_conv_update_runtime.py', 'intent_v2_triton_causal_conv_update_runtime')
    source_module = runtime.RUNTIME.load_causal_conv(context.project_root / 'corpus/gpu/baselines/triton/fla/conv/causal1d/ops.py')
    source_state = initial.clone()
    state: dict[str, object] = {}

    def source_launch():
        (output, updated) = source_module.causal_conv1d_update(x, source_state, weight=weight, bias=bias, activation='silu')
        state['output'] = output
        state['updated'] = updated
    source_launch()
    source = PreparedCall(launch=source_launch, result=lambda : (state['output'], state['updated']), reset=lambda : source_state.copy_(initial))
    return checked_call(source, (Tolerance(atol=0.05, rtol=0.05), Tolerance(atol=0.0)), cuda_graph=False, owners=locals())
