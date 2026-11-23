from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from triton.runtime.errors import OutOfResources
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import ComparisonUnavailable
from intent_bench.implementations.context import Tolerance

def prepare(context):
    from .attention_inputs import backward, backward_source
    q,k,v,output,grad_output,lse,scale = context.inputs if context.inputs is not None else backward().arguments
    source_module = backward_source(context.project_root)
    source = functional_launch(lambda : source_module.flash_bwd(q, k, v, output, lse, grad_output))
    return checked_call(source, (Tolerance(atol=0.125), Tolerance(atol=0.25), Tolerance(atol=0.125)), cuda_graph=False, note='source 修正了中间量写死 bf16 与原 fp16 输入冲突的 dtype 转换；保持原输入与容差。', owners=locals())

def _prepare_source_launch(launch):
    try:
        launch()
    except OutOfResources as error:
        raise ComparisonUnavailable('source_device_resource_gap', f"vendored Triton source exceeds this device's resources: {error}") from error
