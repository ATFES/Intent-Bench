from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    gradient = torch.randn((8 * 1024 * 1024,), dtype=torch.float32)
    initial = (torch.randn_like(gradient), torch.randn_like(gradient), torch.rand_like(gradient) + 0.5)
    scalars = (0.001, 0.9, 0.999, 1.0 - 0.9 ** 10, 1.0 - 0.999 ** 10, 1e-08, 0.01)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

    def side(function):
        state = tuple((value.clone() for value in initial))

        def prepare():
            for (destination, source) in zip(state, initial):
                destination.copy_(source)
        return PreparedCall(lambda : function(gradient, *state, *scalars), lambda : state, reset=prepare)
    return checked_call(side(runtime.adamw), Tolerance(atol=1e-05, rtol=1e-05), device_type='cpu', note='PyTorch eager CPU AdamW N8388608 f32 reference; independent parameter/moment state restored outside timing.', owners=locals())
