from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    x,dy,weight,mean,rstd,_ = context.inputs
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_layer_norm_backward')
    source_state = {}

    def source_launch():
        source_state['output'] = runtime.layer_norm_backward(x, dy, weight, mean, rstd)
    return checked_call(PreparedCall(source_launch, lambda : source_state['output']), (Tolerance(atol=0.05), Tolerance(atol=0.125), Tolerance(atol=0.125)), cuda_graph=False, device_type='cpu', note='既有B4096-N4096 bf16 LayerNorm backward；rows partial后接reduce，完整返回dx/dweight/dbias，PyTorch CPU reference同算法。', owners=locals())
