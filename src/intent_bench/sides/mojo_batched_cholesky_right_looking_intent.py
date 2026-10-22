from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
import intent
from intent.examples.kernels.variants.decomposition import batched_cholesky_right_looking
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import intent_call
CHOLESKY_BATCH = 256
CHOLESKY_SIZE = 16

def prepare(context):
    seed = torch.randn((CHOLESKY_BATCH, CHOLESKY_SIZE, CHOLESKY_SIZE), dtype=torch.float32)
    initial = seed @ seed.transpose(1, 2)
    initial.add_(torch.eye(CHOLESKY_SIZE)[None], alpha=CHOLESKY_SIZE)
    generated_matrix = initial.clone()
    artifact = context.compile(batched_cholesky_right_looking)

    generated = intent_call(artifact, (generated_matrix,), observe=lambda _: generated_matrix,
                            reset=lambda: generated_matrix.copy_(initial))
    return checked_call(generated, Tolerance(atol=0.0003), device_type='cpu', note='Prepared B256/N16 f32 Cholesky; original left-looking versus right-looking semantics. InOut restoration outside timing.', owners=locals())
