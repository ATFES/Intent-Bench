from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.optimization.adafactor import adafactor_update_rows, adafactor_update_columns, adafactor_apply
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    gradient = torch.randn((4096, 4096), dtype=torch.float32) * 0.01
    initial = (torch.randn_like(gradient), torch.rand((4096,)) * 0.01, torch.rand((4096,)) * 0.01, torch.zeros((1,)))
    (decay, learning_rate, epsilon) = (0.8, 0.01, 1e-08)
    (rows, columns, apply) = (context.compile(definition) for definition in (adafactor_update_rows, adafactor_update_columns, adafactor_apply))

    def pipeline(gradient, parameter, row_state, column_state, row_mean, decay, learning_rate, epsilon):
        rows.run(gradient, row_state, row_mean, decay, 1.0 / 4096, 1.0 / 4096)
        columns.run(gradient, column_state, decay, 1.0 / 4096)
        apply.run(gradient, row_state, column_state, row_mean, parameter, learning_rate, epsilon)

    def side(function):
        state = tuple((value.clone() for value in initial))

        def prepare():
            for (destination, source) in zip(state, initial):
                destination.copy_(source)
        sequence = IntentSequence()
        parameter, row_state, column_state, row_mean = state
        sequence.add(rows, gradient, row_state, row_mean, decay, 1.0 / 4096, 1.0 / 4096)
        sequence.add(columns, gradient, column_state, decay, 1.0 / 4096)
        sequence.add(apply, gradient, row_state, column_state, row_mean, parameter, learning_rate, epsilon)
        return sequence.prepared(lambda: state, reset=prepare,
            host_run=lambda: function(gradient, *state, decay, learning_rate, epsilon))
    return checked_call(side(pipeline), (Tolerance(atol=2e-05), Tolerance(atol=2e-07), Tolerance(atol=2e-07), Tolerance(atol=2e-07)), cuda_graph=False, device_type='cpu', note='既有4096x4096 f32 Adafactor三kernel完整调用，含原子row mean和parameter/row/column状态；单NUMA8核，PyTorch CPU reference；每次恢复初始状态且恢复不计时。', owners=locals())
