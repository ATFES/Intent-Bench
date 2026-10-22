from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.optimization.adafactor import adafactor_update_rows, adafactor_update_columns
from intent.examples.kernels.variants.decomposition import adafactor_apply_scalar_product
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    _inlined_adafactor_gradient = torch.randn((4096, 4096), dtype=torch.float32) * 0.01
    _inlined_adafactor_initial = (torch.randn_like(_inlined_adafactor_gradient), torch.rand((4096,)) * 0.01, torch.rand((4096,)) * 0.01, torch.zeros((1,)))
    (_inlined_adafactor_decay, _inlined_adafactor_learning_rate, _inlined_adafactor_epsilon) = (0.8, 0.01, 1e-08)
    (_inlined_adafactor_rows, _inlined_adafactor_columns, _inlined_adafactor_apply) = (context.compile(_inlined_adafactor_definition) for _inlined_adafactor_definition in (adafactor_update_rows, adafactor_update_columns, adafactor_apply_scalar_product))

    def _inlined_adafactor_pipeline(_inlined_adafactor_gradient, _inlined_adafactor_parameter, _inlined_adafactor_row_state, _inlined_adafactor_column_state, _inlined_adafactor_row_mean, _inlined_adafactor_decay, _inlined_adafactor_learning_rate, _inlined_adafactor_epsilon):
        _inlined_adafactor_rows.run(_inlined_adafactor_gradient, _inlined_adafactor_row_state, _inlined_adafactor_row_mean, _inlined_adafactor_decay, 1.0 / 4096, 1.0 / 4096)
        _inlined_adafactor_columns.run(_inlined_adafactor_gradient, _inlined_adafactor_column_state, _inlined_adafactor_decay, 1.0 / 4096)
        _inlined_adafactor_apply.run(_inlined_adafactor_gradient, _inlined_adafactor_row_state, _inlined_adafactor_column_state, _inlined_adafactor_row_mean, _inlined_adafactor_parameter, _inlined_adafactor_learning_rate, _inlined_adafactor_epsilon)

    def _inlined_adafactor_side(_inlined_adafactor_function):
        _inlined_adafactor_state = tuple((_inlined_adafactor_value.clone() for _inlined_adafactor_value in _inlined_adafactor_initial))

        def _inlined_adafactor_prepare():
            for (_inlined_adafactor_destination, _inlined_adafactor_source) in zip(_inlined_adafactor_state, _inlined_adafactor_initial):
                _inlined_adafactor_destination.copy_(_inlined_adafactor_source)
        sequence = IntentSequence()
        parameter, row_state, column_state, row_mean = _inlined_adafactor_state
        sequence.add(_inlined_adafactor_rows, _inlined_adafactor_gradient, row_state, row_mean, _inlined_adafactor_decay, 1.0 / 4096, 1.0 / 4096)
        sequence.add(_inlined_adafactor_columns, _inlined_adafactor_gradient, column_state, _inlined_adafactor_decay, 1.0 / 4096)
        sequence.add(_inlined_adafactor_apply, _inlined_adafactor_gradient, row_state, column_state, row_mean, parameter, _inlined_adafactor_learning_rate, _inlined_adafactor_epsilon)
        return sequence.prepared(lambda: _inlined_adafactor_state, reset=_inlined_adafactor_prepare,
            host_run=lambda: _inlined_adafactor_function(_inlined_adafactor_gradient, *_inlined_adafactor_state, _inlined_adafactor_decay, _inlined_adafactor_learning_rate, _inlined_adafactor_epsilon))
    return checked_call(_inlined_adafactor_side(_inlined_adafactor_pipeline), (Tolerance(atol=2e-06), Tolerance(atol=2e-07), Tolerance(atol=2e-07), Tolerance(atol=2e-07)), cuda_graph=False, device_type='cpu', note='既有4096x4096 f32 Adafactor三kernel完整调用，含原子row mean和parameter/row/column状态；单NUMA8核，PyTorch CPU reference；每次恢复初始状态且恢复不计时。', owners=locals())
