from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.variants.decomposition import adamw_update_moments, adamw_update_parameter
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    gradient = torch.randn((8 * 1024 * 1024,), dtype=torch.float32) * 0.01
    initial = (torch.randn_like(gradient), torch.randn_like(gradient) * 0.01, torch.rand_like(gradient) * 0.01)
    (moments, parameter) = (intent.compile(definition, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options) for definition in (adamw_update_moments, adamw_update_parameter))

    def pipeline(value, first, second):
        moments.run(gradient, first, second, 0.9, 0.999)
        parameter.run(value, first, second, 0.001, 10.0, 1000.0, 1e-08, 0.01)

    def side(function):
        state = tuple((value.clone() for value in initial))

        def prepare():
            for (destination, source) in zip(state, initial):
                destination.copy_(source)
        return PreparedLaunch(lambda : function(*state), lambda : state, prepare=prepare)
    return Side(side(pipeline), Tolerance(atol=2e-06), cuda_graph=False, device_type='cpu', note='原N8388608 f32两阶段AdamW变体输入和2e-6容差；moments后更新parameter，variant接收inverse bias10/1000，除法CPUreference接收对应correction0.1/0.001；独立状态恢复不计时，计两kernel完整host调用。', owners=locals())
