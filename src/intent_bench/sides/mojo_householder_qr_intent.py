from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.factorization.householder_qr import batched_householder_qr
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall
from intent_bench.observed import with_writes

def prepare(context):
    matrices = torch.randn((128, 32, 16), dtype=torch.float32)
    _inlined_factorization_artifact = context.compile(batched_householder_qr)

    def _inlined_factorization_side(_inlined_factorization_function):
        _inlined_factorization_matrices = matrices.clone()
        _inlined_factorization_result = {}

        def _inlined_factorization_launch():
            _inlined_factorization_result['output'] = _inlined_factorization_function(_inlined_factorization_matrices)
        return PreparedCall(_inlined_factorization_launch, lambda : _inlined_factorization_result['output'], reset=lambda : _inlined_factorization_matrices.copy_(matrices))
    return checked_call(_inlined_factorization_side(with_writes(_inlined_factorization_artifact)), (Tolerance(atol=0.0003), Tolerance(atol=0.0003)), cuda_graph=False, device_type='cpu', note='既有batched factorization相同算法/输入规模/f32；单NUMA8核，PyTorch CPU reference写回相同InOut ABI；双方完整host调用，含内部临时量与输出分配，每次恢复输入矩阵且恢复不计时。', owners=locals())
