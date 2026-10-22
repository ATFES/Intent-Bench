from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.compaction.unique_consecutive import unique_consecutive_rows
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import intent_call
ROWS = 64
VALUES = 4096

def prepare(context):
    increments = torch.randint(0, 5, (ROWS, VALUES), dtype=torch.int32)
    values = torch.cumsum(increments == 0, dim=1, dtype=torch.int32)
    generated_values = values.clone()
    generated_lengths = torch.zeros_like(generated_values)
    artifact = context.compile(unique_consecutive_rows)
    def generated_outputs(output):
        (unique_values, inverse, counts) = output
        return (unique_values, inverse, counts, generated_lengths)
    generated = intent_call(artifact, (generated_values, generated_lengths), observe=generated_outputs,
                            reset=lambda: generated_lengths.zero_())
    return checked_call(generated, tuple((Tolerance(atol=0.0) for _ in range(4))), device_type='cpu', note='Prepared unique-consecutive row-wise run-length encoding; independent i32 InOut run_lengths.', owners=locals())
