from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    gradient = torch.randn((4096, 4096), dtype=torch.float32) * 0.01
    initial = (torch.randn_like(gradient), torch.rand((4096,)) * 0.01, torch.rand((4096,)) * 0.01, torch.zeros((1,)))
    (decay, learning_rate, epsilon) = (0.8, 0.01, 1e-08)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

    def side(function):
        state = tuple((value.clone() for value in initial))

        def prepare():
            for (destination, source) in zip(state, initial):
                destination.copy_(source)
        return PreparedLaunch(lambda : function(gradient, *state, decay, learning_rate, epsilon), lambda : state, prepare=prepare)
    return Side(side(runtime.adafactor), (Tolerance(atol=2e-05), Tolerance(atol=2e-07), Tolerance(atol=2e-07), Tolerance(atol=2e-07)), cuda_graph=False, device_type='cpu', note='既有4096x4096 f32 Adafactor三kernel完整调用，含原子row mean和parameter/row/column状态；单NUMA8核，PyTorch CPU reference；每次恢复初始状态且恢复不计时。', owners=locals())
