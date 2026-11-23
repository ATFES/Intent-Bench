from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.implementations.context import Tolerance

def prepare(context):
    from .attention_inputs import backward
    q,k,v,output,grad_output,lse,scale = context.inputs if context.inputs is not None else backward().arguments
    root = context.project_root / "corpus/gpu/baselines/triton/meta-applied-ai"
    runtime = load_module(root / "support/runtime.py", "intent_v2_triton_flash_backward_runtime")
    source_module = runtime.load_source(root / "attention/flash_backward/flash_backward.py",
                                        "intent_v2_triton_flash_backward_source")
    source = functional_launch(lambda : source_module.flash_bwd(q, k, v, output, lse, grad_output))
    return checked_call(source, (Tolerance(atol=0.125), Tolerance(atol=0.25), Tolerance(atol=0.125)), cuda_graph=False, note='source 修正了中间量写死 bf16 与原 fp16 输入冲突的 dtype 转换；保持原输入与容差。', owners=locals())
