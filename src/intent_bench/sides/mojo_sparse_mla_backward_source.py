from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    (batch, sequence, key_value_sequence) = (1, 4096, 8192)
    (heads, key_value_groups) = (64, 1)
    (query_dimension, value_dimension, topk) = (576, 512, 2048)
    query = torch.randn((batch, sequence, heads, query_dimension), dtype=torch.bfloat16)
    key_value = torch.randn((batch, key_value_sequence, key_value_groups, query_dimension), dtype=torch.bfloat16)
    grad_output = torch.randn((batch, sequence, heads, value_dimension), dtype=torch.bfloat16)
    positions = torch.arange(sequence, dtype=torch.int32)
    offsets = torch.arange(topk, dtype=torch.int32)
    causal_indices = positions[:, None] - offsets[None, :] - 1
    causal_indices = torch.where(offsets[None, :] < positions[:, None], causal_indices, torch.full_like(causal_indices, key_value_sequence))
    selected_indices = causal_indices.view(batch, sequence, key_value_groups, topk).contiguous()
    scale = query_dimension ** (-0.5)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_sparse_mla_backward')
    (output, lse, source_backward) = runtime.prepare_sparse_mla_backward(query, key_value, grad_output, selected_indices, scale)
    source_state = {}

    def source_launch():
        source_state['output'] = source_backward()
    return Side(PreparedLaunch(source_launch, lambda : source_state['output']), (Tolerance(atol=0.25, rtol=0.05), Tolerance(atol=0.25, rtol=0.05)), cuda_graph=False, device_type='cpu', note='既有B1-S4096-SKV8192-H64-DQ576-DV512-topk2048 bf16 sparse MLA backward；计入 f32 gradKV workspace 清零及 delta/main/gradKV cast 三kernel；CPU reference 同样重算概率并保持 bf16 dScore/P 转换，前向 output/LSE 准备不计入反向时间。', owners=locals())
