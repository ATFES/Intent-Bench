from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.references.torch_ops import triangular_solve
from intent_bench.implementations.context import Tolerance
from intent_bench.source_support import reference_launch

def prepare(context):
    from intent_bench.trees import to_device
    lower, initial = to_device(context.inputs, 'cuda')
    return checked_call(reference_launch(lambda : triangular_solve(*(lower, initial.clone()))), Tolerance(atol=0.0001, rtol=0.0001), device_type='cuda', owners=locals())
