from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import intent
import torch
from intent.examples.kernels.factorization.householder_qr import batched_householder_qr
from intent_bench.implementations.context import Tolerance

def prepare(context):
    matrices = torch.randn((128, 32, 16), dtype=torch.float32)
    _inlined_factorization_artifact = context.compile(batched_householder_qr)

    def _inlined_factorization_side():
        _inlined_factorization_matrices = matrices.clone()
        return intent_call(_inlined_factorization_artifact, (_inlined_factorization_matrices,),
                           reset=lambda: _inlined_factorization_matrices.copy_(matrices),
                           observe=lambda tau: (_inlined_factorization_matrices, tau))
    return checked_call(_inlined_factorization_side(), (Tolerance(atol=0.0003), Tolerance(atol=0.0003)), cuda_graph=False, device_type='cpu', note='既有batched factorization相同算法/输入规模/f32；单NUMA8核，PyTorch CPU reference写回相同InOut ABI；Intent计prepared native调用，内部临时量仍计入，每次恢复输入矩阵且恢复不计时。', owners=locals())
