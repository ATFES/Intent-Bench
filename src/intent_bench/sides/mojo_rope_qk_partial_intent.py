from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
import intent
from intent.examples.kernels.position.rope import rotary_qk_partial_inplace
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import intent_call
PARTIAL_ROTARY_DIMENSION = 64

def prepare(context):
    (batch, sequence, query_heads, kv_heads, dimension) = (4, 2048, 32, 8, 128)
    rotary_dimension = PARTIAL_ROTARY_DIMENSION
    half = rotary_dimension // 2
    query_initial = torch.randn((batch, query_heads, sequence, dimension), dtype=torch.float16) * 0.1
    key_initial = torch.randn((batch, kv_heads, sequence, dimension), dtype=torch.float16) * 0.1
    angles = torch.randn((1, sequence, half), dtype=torch.float32)
    cosine_half = angles.cos().to(torch.float16)
    sine_half = angles.sin().to(torch.float16)
    cosine = torch.cat((cosine_half, cosine_half), dim=2)
    sine = torch.cat((sine_half, sine_half), dim=2)
    artifact = context.compile(rotary_qk_partial_inplace)
    generated_query = query_initial.clone()
    generated_key = key_initial.clone()

    def restore_generated():
        generated_query.copy_(query_initial)
        generated_key.copy_(key_initial)
    generated = intent_call(artifact, (generated_query, generated_key, cosine, sine),
                            observe=lambda _: (generated_query, generated_key), reset=restore_generated)
    return checked_call(generated, (Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.002, rtol=0.002)), cuda_graph=False, device_type='cpu', note='既有 partial Q/K rotary f16 InOut；仅旋转前64维，PyTorch CPU reference 保持原f32乘法后f16写回；每次恢复 query/key，恢复不计时；默认计准备后的 native 执行；Python 调用仅在 host_call 模式计时。', owners=locals())
