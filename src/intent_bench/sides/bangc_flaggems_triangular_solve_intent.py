from __future__ import annotations
from intent_bench.sides import Side
import torch
from kernels.factorization.triangular_solve import batched_lower_triangular_solve
from intent_bench.corpus._common.model import Tolerance
from intent_bench.implementations.bangc_sequence import RemoteSequence
from intent_bench.implementations.bangc import prepare_sequence

def prepare(context):
    (batch, size) = (4096, 16)
    lower = torch.tril(torch.randn((batch, size, size), device='cuda', dtype=torch.float32))
    lower.diagonal(dim1=-2, dim2=-1).add_(2.0)
    initial = torch.randn((batch, size), device='cuda', dtype=torch.float32)
    solution = initial.clone()
    sequence = RemoteSequence(context)
    sequence.add(batched_lower_triangular_solve, {'lower': lower, 'solution': solution})
    return Side(prepare_sequence(sequence, solution), Tolerance(atol=0.0001, rtol=0.0001), device_type='mlu', owners=locals())
