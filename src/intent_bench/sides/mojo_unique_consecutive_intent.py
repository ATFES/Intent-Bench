from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.compaction.unique_consecutive import unique_consecutive_rows
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
ROWS = 64
VALUES = 4096

def prepare(context):
    increments = torch.randint(0, 5, (ROWS, VALUES), dtype=torch.int32)
    values = torch.cumsum(increments == 0, dim=1, dtype=torch.int32)
    generated_values = values.clone()
    generated_lengths = torch.zeros_like(generated_values)
    artifact = intent.compile(unique_consecutive_rows, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    generated_state = {}

    def generated_launch():
        generated_state['output'] = artifact.run(generated_values, generated_lengths)

    def generated_outputs():
        (unique_values, inverse, counts) = generated_state['output']
        return (unique_values, inverse, counts, generated_lengths)
    return Side(PreparedLaunch(generated_launch, generated_outputs, prepare=lambda : generated_lengths.zero_()), tuple((Tolerance(atol=0.0) for _ in range(4))), cuda_graph=False, device_type='cpu', note='既有 unique-consecutive row-wise run-length encoding；输入/输出均 i32，生成端与 CPU reference 使用独立 InOut run_lengths。', owners=locals())
