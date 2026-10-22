from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
import intent
from intent.examples.kernels.position.rope import rotary_qk_inplace
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import intent_call

def prepare(context):
    (batch, sequence, query_heads, kv_heads, dimension) = (4, 4096, 32, 8, 128)
    query_initial = torch.randn((batch, query_heads, sequence, dimension), dtype=torch.float16)
    key_initial = torch.randn((batch, kv_heads, sequence, dimension), dtype=torch.float16)
    positions = torch.arange(sequence, dtype=torch.float32)
    inverse = 1.0 / 10000 ** (torch.arange(0, dimension, 2, dtype=torch.float32) / dimension)
    frequency = torch.outer(positions, inverse)
    embedding = torch.cat((frequency, frequency), dim=-1)
    cosine = embedding.cos().unsqueeze(0).to(torch.float16)
    sine = embedding.sin().unsqueeze(0).to(torch.float16)
    artifact = context.compile(rotary_qk_inplace)
    generated_query = query_initial.clone()
    generated_key = key_initial.clone()

    def restore_generated():
        generated_query.copy_(query_initial)
        generated_key.copy_(key_initial)
    generated = intent_call(artifact, (generated_query, generated_key, cosine, sine),
                            observe=lambda _: (generated_query, generated_key), reset=restore_generated)
    return checked_call(generated, (Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.002, rtol=0.002)), cuda_graph=False, device_type='cpu', note='既有 rope_qk f16 InOut；PyTorch CPU reference，单 NUMA 8 核；每次调用前恢复 query/key，恢复不计时；默认计准备后的 native 执行；Python 调用仅在 host_call 模式计时。', owners=locals())
