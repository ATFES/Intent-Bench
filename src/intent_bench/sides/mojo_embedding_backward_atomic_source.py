from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def prepare(context):
    (tokens, vocabulary, features) = (32768, 8192, 1021)
    token_ids = torch.arange(tokens, dtype=torch.int32)
    indices = token_ids * 17 % vocabulary
    grad_output = torch.randn((tokens, features), dtype=torch.float32) * 0.001
    initial_weight = torch.zeros((vocabulary, features), dtype=torch.float32)
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_embedding_backward_atomic')
    source_weight = initial_weight.clone()

    def restore_source():
        source_weight.copy_(initial_weight)
    source = PreparedCall(lambda : runtime.embedding_backward_atomic(indices, grad_output, source_weight), lambda : source_weight, reset=restore_source)
    return checked_call(source, Tolerance(atol=2e-06, rtol=0.0), cuda_graph=False, device_type='cpu', note='既有 embedding backward atomic M32768/V8192/D1021 f32 InOut；PyTorch index_add_ reference，单 NUMA 8 核；每次调用前恢复 zero weight，恢复不计时；双方计完整 host 调用。', owners=locals())
