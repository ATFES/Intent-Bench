from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.backward.embedding import embedding_forward_lookup_bf16
from intent_bench.corpus._common.measurement import compile_single
from intent_bench.corpus._common.model import PreparedLaunch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, sequence, vocabulary, hidden) = (8, 2048, 32768, 4096)
    table = torch.randn((vocabulary, hidden), device='cuda', dtype=torch.bfloat16)
    indices = torch.randint(0, vocabulary, (batch, sequence), device='cuda', dtype=torch.int64)
    (_, generated_base) = compile_single(context, embedding_forward_lookup_bf16, (table, indices.reshape(-1)))
    generated = PreparedLaunch(launch=generated_base.launch, outputs=lambda : generated_base.outputs().view(batch, sequence, hidden))
    return Side(generated, Tolerance(atol=0.0), cuda_graph=True, owners=locals())
