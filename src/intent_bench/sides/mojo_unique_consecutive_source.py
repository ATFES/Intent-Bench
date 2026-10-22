from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall
ROWS = 64
VALUES = 4096

def prepare(context):
    increments = torch.randint(0, 5, (ROWS, VALUES), dtype=torch.int32)
    values = torch.cumsum(increments == 0, dim=1, dtype=torch.int32)
    source_values = values.clone()
    source_lengths = torch.zeros_like(source_values)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_unique_consecutive')
    source_state = {}

    def source_launch():
        source_state['output'] = runtime.unique_consecutive(source_values, source_lengths)

    def source_outputs():
        return (*source_state['output'], source_lengths)
    return checked_call(PreparedCall(source_launch, source_outputs, reset=lambda : source_lengths.zero_()), tuple((Tolerance(atol=0.0) for _ in range(4))), cuda_graph=False, device_type='cpu', note='既有 unique-consecutive row-wise run-length encoding；输入/输出均 i32，生成端与 CPU reference 使用独立 InOut run_lengths。', owners=locals())
