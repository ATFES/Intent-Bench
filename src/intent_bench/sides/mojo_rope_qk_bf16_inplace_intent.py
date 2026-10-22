from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
import intent
from intent.examples.kernels.position.rope import rotary_qk_bf16_inplace
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import intent_call

def prepare(context):
    (batch, sequence, query_heads, kv_heads, dimension) = (2, 4096, 32, 8, 128)
    query_initial = torch.randn((batch, query_heads, sequence, dimension), dtype=torch.bfloat16)
    key_initial = torch.randn((batch, kv_heads, sequence, dimension), dtype=torch.bfloat16)
    cosine = torch.randn((1, sequence, dimension), dtype=torch.bfloat16)
    sine = torch.randn_like(cosine)
    artifact = context.compile(rotary_qk_bf16_inplace)
    generated_query = query_initial.clone()
    generated_key = key_initial.clone()

    def restore_generated():
        generated_query.copy_(query_initial)
        generated_key.copy_(key_initial)
    generated = intent_call(artifact, (generated_query, generated_key, cosine, sine),
                            observe=lambda _: (generated_query, generated_key), reset=restore_generated)
    return checked_call(generated, (Tolerance(atol=0.02, rtol=0.01), Tolerance(atol=0.02, rtol=0.01)), cuda_graph=False, device_type='cpu', note='既有 Q/K rotary bf16 InOut；CPU reference 按 Intent 显式 f32 中间运算后转回 bf16，与原 CUDA source 的 bf16 中间舍入不同；每次恢复 query/key，恢复不计时；默认计准备后的 native 执行；Python 调用仅在 host_call 模式计时。', owners=locals())
