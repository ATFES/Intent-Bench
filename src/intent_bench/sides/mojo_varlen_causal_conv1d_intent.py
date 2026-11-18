from __future__ import annotations
from intent_bench.implementations.prepared import checked_call, intent_call
import torch
from intent.examples.programs.structured import VarlenCausalConvolution
from intent_bench.implementations.context import Tolerance

def prepare(context):
    lengths = (2048, 1536, 1024, 512)
    offsets = torch.tensor((0, 2048, 3584, 4608, 5120), dtype=torch.int64)
    chunks = torch.tensor([(sequence, chunk) for (sequence, length) in enumerate(lengths) for chunk in range((length + 63) // 64)], dtype=torch.int32)
    x = torch.randn((sum(lengths), 4096), dtype=torch.bfloat16)
    weight = torch.randn((4096, 4), dtype=torch.float32)
    bias = torch.randn((4096,), dtype=torch.float32)
    program = VarlenCausalConvolution.from_jit(context.compile)
    generated = intent_call(program, (x, offsets, chunks, weight, bias))
    return checked_call(generated, (Tolerance(atol=0.05, rtol=0.05), Tolerance(atol=0.0)),
                        device_type='cpu', owners=locals())
