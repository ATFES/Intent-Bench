from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (tokens, hidden, output, experts, topk) = (2048, 4096, 14336, 8, 2)
    x = torch.randn((tokens, hidden), device='cuda', dtype=torch.bfloat16)
    weight = torch.randn((experts, output, hidden), device='cuda', dtype=torch.bfloat16)
    ids = torch.stack((torch.arange(tokens, device='cuda') % experts, (torch.arange(tokens, device='cuda') + 1) % experts), dim=1).to(torch.int32)
    projection_runtime = load_module(context.project_root / 'corpus/gpu/baselines/triton/meta-applied-ai/moe/support/projection_runtime.py', 'intent_v2_triton_moe_projection_runtime')
    runtime = projection_runtime._runtime_module()
    if 'grouped' == 'grouped':
        source_path = 'corpus/gpu/baselines/triton/meta-applied-ai/moe/grouped/v0_moe_fused.py'
        config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}
        block_m = config['BLOCK_SIZE_M']
    elif 'grouped' == 'splitk':
        source_path = 'corpus/gpu/baselines/triton/meta-applied-ai/moe/splitk/v1_moe_fused.py'
        config = {'block_m': 32, 'block_n': 64, 'block_k': 64, 'group_m': 8, 'split_k': 2}
        block_m = config['block_m']
    else:
        raise ValueError('grouped')
    source_module = runtime.load_source(context.project_root / source_path, f"intent_v2_triton_moe_{'grouped'}_source", stub_vllm=True)
    (sorted_ids, expert_ids, padded) = projection_runtime._balanced_metadata(ids, block_m, experts)
    routed_weight = torch.full((tokens, topk), 0.5, device='cuda', dtype=torch.float32)
    source_output = torch.zeros((tokens, topk, output), device='cuda', dtype=torch.bfloat16)

    def launch():
        source_module.invoke_fused_moe_kernel(x, weight, source_output, routed_weight, ids, sorted_ids, expert_ids, padded, False, topk, config)
    launch()
    source = PreparedLaunch(launch=launch, outputs=lambda : source_output, prepare=source_output.zero_ if 'grouped' == 'splitk' else None)
    return Side(source, Tolerance(atol=2.5 if 'grouped' == 'splitk' else 0.1, rtol=0.05), cuda_graph=False, owners=locals())
