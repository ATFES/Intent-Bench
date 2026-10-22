from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent.examples.kernels.ragged.grouped_gemm import routed_expert_projection_bf16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.implementations.context import Tolerance

def prepare(context):
    (tokens, hidden, output, experts, topk) = (2048, 4096, 14336, 8, 2)
    x = torch.randn((tokens, hidden), device='cuda', dtype=torch.bfloat16)
    weight = torch.randn((experts, output, hidden), device='cuda', dtype=torch.bfloat16)
    ids = torch.stack((torch.arange(tokens, device='cuda') % experts, (torch.arange(tokens, device='cuda') + 1) % experts), dim=1).to(torch.int32)
    flat_ids = ids.reshape(-1)
    member_routes = torch.argsort(flat_ids, stable=True).to(torch.int32)
    counts = torch.bincount(flat_ids, minlength=experts)
    expert_offsets = torch.cat((torch.zeros(1, device='cuda', dtype=torch.int64), counts.cumsum(0))).to(torch.int32)
    (_, generated) = compile_single(context, routed_expert_projection_bf16, (x, expert_offsets, member_routes, weight), constexprs={'TOP_K': topk})
    return checked_call(generated, Tolerance(atol=2.5 if 'grouped' == 'splitk' else 0.1, rtol=0.05), cuda_graph=False, owners=locals())
