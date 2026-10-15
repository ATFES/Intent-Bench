from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
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
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference_rope_qk_partial')
    source_query = query_initial.clone()
    source_key = key_initial.clone()

    def source_launch():
        runtime.rope_qk_partial(source_query, source_key, cosine, sine)

    def restore_source():
        source_query.copy_(query_initial)
        source_key.copy_(key_initial)
    source = PreparedLaunch(source_launch, lambda : (source_query, source_key), prepare=restore_source)
    return Side(source, (Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.002, rtol=0.002)), cuda_graph=False, device_type='cpu', note='既有 partial Q/K rotary f16 InOut；仅旋转前64维，PyTorch CPU reference 保持原f32乘法后f16写回；每次恢复 query/key，恢复不计时；双方计完整host调用。', owners=locals())
