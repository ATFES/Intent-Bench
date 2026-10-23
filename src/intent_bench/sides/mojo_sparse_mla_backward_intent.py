from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, IntentSequence
import intent
import torch
from intent.examples.kernels.backward.sparse_mla import sparse_mla_backward_delta, sparse_mla_backward_main, sparse_mla_grad_kv_cast
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance

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
    delta_artifact = context.compile(sparse_mla_backward_delta)
    constexprs = {'HEAD_GROUP': heads // key_value_groups}
    main_artifact = context.compile(sparse_mla_backward_main, constexprs=constexprs)
    cast_artifact = context.compile(sparse_mla_grad_kv_cast)
    delta = torch.empty((batch, sequence, heads), dtype=torch.float32)
    grad_key_value = torch.zeros_like(key_value, dtype=torch.float32)
    grad_query = torch.empty_like(query)
    grad_key_value_bf16 = torch.empty_like(key_value)

    def generated_launch():
        grad_key_value.zero_()
        delta_artifact(output, grad_output, delta)
        main_artifact(query, key_value, grad_output, selected_indices, lse, delta, grad_key_value, grad_query, scale)
        cast_artifact(grad_key_value, grad_key_value_bf16)
    sequence = IntentSequence()
    sequence.add(delta_artifact, output, grad_output, outputs=(delta,))
    sequence.add(main_artifact, query, key_value, grad_output, selected_indices, lse, delta,
                 grad_key_value, scale, outputs=(grad_query,))
    sequence.add(cast_artifact, grad_key_value, outputs=(grad_key_value_bf16,))
    generated = sequence.prepared(lambda: (grad_query, grad_key_value_bf16), host_run=generated_launch,
                                  initializations=((grad_key_value, 0),))
    return checked_call(generated, (Tolerance(atol=0.25, rtol=0.05), Tolerance(atol=0.25, rtol=0.05)), cuda_graph=False, device_type='cpu', note='既有B1-S4096-SKV8192-H64-DQ576-DV512-topk2048 bf16 sparse MLA backward；f32 gradKV workspace 清零及 delta/main/gradKV cast 三kernel在同一native计时间隔内；CPU reference 同样重算概率并保持 bf16 dScore/P 转换，前向 output/LSE 准备不计入反向时间。', owners=locals())
