from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.optimization.adamw import adamw_update
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    gradient = torch.randn((8 * 1024 * 1024,), dtype=torch.float32)
    initial = (torch.randn_like(gradient), torch.randn_like(gradient), torch.rand_like(gradient) + 0.5)
    scalars = (0.001, 0.9, 0.999, 1.0 - 0.9 ** 10, 1.0 - 0.999 ** 10, 1e-08, 0.01)
    artifact = intent.compile(adamw_update, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    generated = torch.compile(artifact.as_torch_op('intent_cpu_benchmark::adamw'), fullgraph=True)

    def side(function):
        state = tuple((value.clone() for value in initial))

        def prepare():
            for (destination, source) in zip(state, initial):
                destination.copy_(source)
        return PreparedLaunch(lambda : function(gradient, *state, *scalars), lambda : state, prepare=prepare)
    return Side(side(generated), Tolerance(atol=1e-05, rtol=1e-05), cuda_graph=False, device_type='cpu', note='既有 AdamW N8388608 f32；Intent 通过 mutable custom op 与 torch.compile(fullgraph=True) 调用；PyTorch eager CPU reference，单 NUMA 8 核；每次调用前恢复相同的 parameter/moments，恢复不计时；双方计完整 host 调用。', owners=locals())
