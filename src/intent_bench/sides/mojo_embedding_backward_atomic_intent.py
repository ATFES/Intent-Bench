from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.backward.embedding import embedding_backward_atomic
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
from intent_bench.cpu_support import report_selection

def prepare(context):
    (tokens, vocabulary, features) = (32768, 8192, 1021)
    token_ids = torch.arange(tokens, dtype=torch.int32)
    indices = token_ids * 17 % vocabulary
    grad_output = torch.randn((tokens, features), dtype=torch.float32) * 0.001
    initial_weight = torch.zeros((vocabulary, features), dtype=torch.float32)
    artifact = intent.compile(embedding_backward_atomic, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    generated_weight = initial_weight.clone()

    def restore_generated():
        generated_weight.copy_(initial_weight)
    generated = PreparedLaunch(lambda : artifact.run(indices, grad_output, generated_weight), lambda : selected_outputs(artifact, generated_weight), prepare=restore_generated)
    return Side(generated, Tolerance(atol=2e-06, rtol=0.0), cuda_graph=False, device_type='cpu', note='既有 embedding backward atomic M32768/V8192/D1021 f32 InOut；PyTorch index_add_ reference，单 NUMA 8 核；每次调用前恢复 zero weight，恢复不计时；双方计完整 host 调用。', owners=locals())

def selected_outputs(artifact, outputs):
    report_selection(artifact.observation)
    return outputs
