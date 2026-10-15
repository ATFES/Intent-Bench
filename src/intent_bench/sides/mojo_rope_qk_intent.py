from __future__ import annotations
from intent_bench.sides import Side
import torch
import intent
from kernels.position.rope import rotary_qk_inplace
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

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
    artifact = intent.compile(rotary_qk_inplace, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    generated_query = query_initial.clone()
    generated_key = key_initial.clone()

    def restore_generated():
        generated_query.copy_(query_initial)
        generated_key.copy_(key_initial)
    generated = PreparedLaunch(lambda : artifact.run(generated_query, generated_key, cosine, sine), lambda : (generated_query, generated_key), prepare=restore_generated)
    return Side(generated, (Tolerance(atol=0.002, rtol=0.002), Tolerance(atol=0.002, rtol=0.002)), cuda_graph=False, device_type='cpu', note='既有 rope_qk f16 InOut；PyTorch CPU reference，单 NUMA 8 核；每次调用前恢复 query/key，恢复不计时；双方计完整 host 调用。', owners=locals())
