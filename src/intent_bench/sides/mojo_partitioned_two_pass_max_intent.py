from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.reduction.two_pass import partitioned_max_partial, partitioned_max_reduce
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    x = -torch.rand((128, 257), dtype=torch.float32)
    partial = context.compile(partitioned_max_partial)
    reduce = context.compile(partitioned_max_reduce)
    state = {}

    def generated_launch():
        workspace = torch.empty((128, 300), dtype=torch.float32)
        partial(x, workspace)
        state['generated'] = reduce.run(workspace)
    sequence = IntentSequence()
    workspace = torch.empty((128, 300), dtype=torch.float32)
    sequence.add(partial, x, outputs=(workspace,))
    output = sequence.add(reduce, workspace)
    call = sequence.prepared(lambda: output, host_run=generated_launch, host_result=lambda: state['generated'])
    return checked_call(call, Tolerance(atol=0.0), device_type='cpu', note='Prepared M128/N257/P300 two-pass max, including empty partitions. Original negative input and exact output; partial fully writes workspace.', owners=locals())
