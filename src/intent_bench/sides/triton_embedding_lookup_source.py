from __future__ import annotations
from intent_bench.sides import Side
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.measurement import functional_launch
from intent_bench.corpus._common.model import Tolerance

def prepare(context):
    (batch, sequence, vocabulary, hidden) = (8, 2048, 32768, 4096)
    table = torch.randn((vocabulary, hidden), device='cuda', dtype=torch.bfloat16)
    indices = torch.randint(0, vocabulary, (batch, sequence), device='cuda', dtype=torch.int64)
    runtime = _runtime(context, 'corpus/gpu/baselines/triton/liger-kernel/embedding/lookup/embedding_runtime.py', 'intent_v2_triton_embedding_lookup')
    source = functional_launch(lambda : runtime.LigerEmbeddingFunction.apply(table, indices))
    return Side(source, Tolerance(atol=0.0), cuda_graph=True, owners=locals())

def _runtime(context, path, name):
    return load_module(context.project_root / path, name)
