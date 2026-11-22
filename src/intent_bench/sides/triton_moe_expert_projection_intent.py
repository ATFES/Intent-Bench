from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.ragged.grouped_gemm import routed_expert_projection_bf16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    x,expert_offsets,member_routes,weight=context.inputs
    topk=member_routes.numel()//x.size(0)
    (_, generated) = compile_single(context, routed_expert_projection_bf16, (x, expert_offsets, member_routes, weight), constexprs={'TOP_K': topk})
    return checked_call(generated, Tolerance(atol=2.5 if 'grouped' == 'splitk' else 0.1, rtol=0.05), cuda_graph=False, owners=locals())
