from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    _inlined_adafactor_gradient = torch.randn((4096, 4096), dtype=torch.float32) * 0.01
    _inlined_adafactor_initial = (torch.randn_like(_inlined_adafactor_gradient), torch.rand((4096,)) * 0.01, torch.rand((4096,)) * 0.01, torch.zeros((1,)))
    (_inlined_adafactor_decay, _inlined_adafactor_learning_rate, _inlined_adafactor_epsilon) = (0.8, 0.01, 1e-08)
    _inlined_adafactor_runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

    def _inlined_adafactor_side(_inlined_adafactor_function):
        _inlined_adafactor_state = tuple((_inlined_adafactor_value.clone() for _inlined_adafactor_value in _inlined_adafactor_initial))

        def _inlined_adafactor_prepare():
            for (_inlined_adafactor_destination, _inlined_adafactor_source) in zip(_inlined_adafactor_state, _inlined_adafactor_initial):
                _inlined_adafactor_destination.copy_(_inlined_adafactor_source)
        return PreparedLaunch(lambda : _inlined_adafactor_function(_inlined_adafactor_gradient, *_inlined_adafactor_state, _inlined_adafactor_decay, _inlined_adafactor_learning_rate, _inlined_adafactor_epsilon), lambda : _inlined_adafactor_state, prepare=_inlined_adafactor_prepare)
    return Side(_inlined_adafactor_side(_inlined_adafactor_runtime.adafactor), (Tolerance(atol=2e-06), Tolerance(atol=2e-07), Tolerance(atol=2e-07), Tolerance(atol=2e-07)), cuda_graph=False, device_type='cpu', note='既有4096x4096 f32 Adafactor三kernel完整调用，含原子row mean和parameter/row/column状态；单NUMA8核，PyTorch CPU reference；每次恢复初始状态且恢复不计时。', owners=locals())
