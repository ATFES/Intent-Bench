from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    gradient = torch.randn((8 * 1024 * 1024,), dtype=torch.float32) * 0.01
    initial = (torch.randn_like(gradient), torch.randn_like(gradient) * 0.01, torch.rand_like(gradient) * 0.01)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

    def reference(value, first, second):
        runtime.adamw(gradient, value, first, second, 0.001, 0.9, 0.999, 0.1, 0.001, 1e-08, 0.01)

    def side(function):
        state = tuple((value.clone() for value in initial))

        def prepare():
            for (destination, source) in zip(state, initial):
                destination.copy_(source)
        return PreparedLaunch(lambda : function(*state), lambda : state, prepare=prepare)
    return Side(side(reference), Tolerance(atol=2e-06), cuda_graph=False, device_type='cpu', note='原N8388608 f32两阶段AdamW变体输入和2e-6容差；moments后更新parameter，variant接收inverse bias10/1000，除法CPUreference接收对应correction0.1/0.001；独立状态恢复不计时，计两kernel完整host调用。', owners=locals())
