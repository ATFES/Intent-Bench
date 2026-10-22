from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.optimization.adamw import adamw_update
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import intent_call

def prepare(context):
    gradient = torch.randn((8 * 1024 * 1024,), dtype=torch.float32)
    initial = (torch.randn_like(gradient), torch.randn_like(gradient), torch.rand_like(gradient) + 0.5)
    scalars = (0.001, 0.9, 0.999, 1.0 - 0.9 ** 10, 1.0 - 0.999 ** 10, 1e-08, 0.01)
    artifact = context.compile(adamw_update)
    state = tuple(value.clone() for value in initial)
    arguments = (gradient, *state, *scalars)
    def restore():
        for destination, source in zip(state, initial):
            destination.copy_(source)
    def host_factory():
        operator = torch.compile(artifact.as_torch_op('intent_cpu_benchmark::adamw'), fullgraph=True)
        return lambda: operator(*arguments)
    generated = intent_call(artifact, arguments, reset=restore, observe=lambda _: state, host_factory=host_factory)
    return checked_call(generated, Tolerance(atol=1e-05, rtol=1e-05), device_type='cpu', note='Prepared native AdamW; state restoration outside timing. torch.compile integration is opt-in via host_call.', owners=locals())
