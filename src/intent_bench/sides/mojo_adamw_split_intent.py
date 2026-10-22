from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.variants.decomposition import adamw_update_moments, adamw_update_parameter
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence

def prepare(context):
    gradient = torch.randn((8 * 1024 * 1024,), dtype=torch.float32) * 0.01
    initial = (torch.randn_like(gradient), torch.randn_like(gradient) * 0.01, torch.rand_like(gradient) * 0.01)
    (moments, parameter) = (context.compile(definition) for definition in (adamw_update_moments, adamw_update_parameter))

    def pipeline(value, first, second):
        moments.run(gradient, first, second, 0.9, 0.999)
        parameter.run(value, first, second, 0.001, 10.0, 1000.0, 1e-08, 0.01)

    def side(function):
        state = tuple((value.clone() for value in initial))

        def prepare():
            for (destination, source) in zip(state, initial):
                destination.copy_(source)
        sequence = IntentSequence()
        value, first, second = state
        sequence.add(moments, gradient, first, second, 0.9, 0.999)
        sequence.add(parameter, value, first, second, 0.001, 10.0, 1000.0, 1e-08, 0.01)
        return sequence.prepared(lambda: state, reset=prepare, host_run=lambda: function(*state))
    return checked_call(side(pipeline), Tolerance(atol=2e-06), cuda_graph=False, device_type='cpu', note='原N8388608 f32两阶段AdamW变体输入和2e-6容差；moments后更新parameter，variant接收inverse bias10/1000，除法CPUreference接收对应correction0.1/0.001；独立状态恢复不计时，默认计两kernel的完整native执行。', owners=locals())
