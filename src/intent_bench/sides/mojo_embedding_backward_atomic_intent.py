from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.backward.embedding import embedding_backward_atomic
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import intent_call
from intent_bench.cpu_support import report_selection

def prepare(context):
    (tokens, vocabulary, features) = (32768, 8192, 1021)
    token_ids = torch.arange(tokens, dtype=torch.int32)
    indices = token_ids * 17 % vocabulary
    grad_output = torch.randn((tokens, features), dtype=torch.float32) * 0.001
    initial_weight = torch.zeros((vocabulary, features), dtype=torch.float32)
    artifact = context.compile(embedding_backward_atomic)
    generated_weight = initial_weight.clone()

    def restore_generated():
        generated_weight.copy_(initial_weight)
    generated = intent_call(artifact, (indices, grad_output, generated_weight), reset=restore_generated,
                            observe=lambda _: selected_outputs(artifact, generated_weight))
    return checked_call(generated, Tolerance(atol=2e-06, rtol=0.0), cuda_graph=False, device_type='cpu', note='既有 embedding backward atomic M32768/V8192/D1021 f32 InOut；PyTorch index_add_ reference，单 NUMA 8 核；每次调用前恢复 zero weight，恢复不计时；默认计准备后的 native 执行；Python 调用仅在 host_call 模式计时。', owners=locals())

def selected_outputs(artifact, outputs):
    report_selection(artifact.observation)
    return outputs
