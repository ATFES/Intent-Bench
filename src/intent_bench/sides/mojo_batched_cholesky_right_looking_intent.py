from __future__ import annotations
from intent_bench.sides import Side
import torch
import intent
from kernels.variants.decomposition import batched_cholesky_right_looking
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
CHOLESKY_BATCH = 256
CHOLESKY_SIZE = 16

def prepare(context):
    seed = torch.randn((CHOLESKY_BATCH, CHOLESKY_SIZE, CHOLESKY_SIZE), dtype=torch.float32)
    initial = seed @ seed.transpose(1, 2)
    initial.add_(torch.eye(CHOLESKY_SIZE)[None], alpha=CHOLESKY_SIZE)
    generated_matrix = initial.clone()
    artifact = intent.compile(batched_cholesky_right_looking, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)

    def generated_launch():
        artifact.run(generated_matrix)
    return Side(PreparedLaunch(generated_launch, lambda : generated_matrix, prepare=lambda : generated_matrix.copy_(initial)), Tolerance(atol=0.0003), cuda_graph=False, device_type='cpu', note='既有 B256-N16 f32 Cholesky left-looking versus right-looking；生成端与 CPU reference 使用独立 InOut 矩阵，每次调用前恢复且恢复不计时；双方计完整 host 调用。', owners=locals())
