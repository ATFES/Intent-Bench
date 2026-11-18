from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import torch
from intent_bench.corpus._common.loading import load_module
from intent_bench.implementations.context import Tolerance
from intent_bench.model import PreparedCall

def _inputs():
    lengths = (2048, 1536, 1024, 512)
    offsets = torch.tensor((0, 2048, 3584, 4608, 5120), dtype=torch.int64)
    x = torch.randn((sum(lengths), 4096), dtype=torch.bfloat16)
    weight = torch.randn((4096, 4), dtype=torch.float32)
    bias = torch.randn((4096,), dtype=torch.float32)
    return x, offsets, weight, bias

def _native(context, operation):
    from intent_bench.implementations.aten_cpu import prepare as native
    arguments = _inputs()
    call = native(operation, arguments, context.environment)
    return checked_call(call, (Tolerance(atol=0.05, rtol=0.05), Tolerance(atol=0.0)),
                        device_type='cpu', owners={'inputs': arguments})

def prepare_native(context):
    return _native(context, 'varlen_unfold')

def prepare_grouped(context):
    return _native(context, 'varlen_causal_conv1d')

def prepare(context):
    x, offsets, weight, bias = _inputs()
    runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

    def side(function):
        state = {}

        def launch():
            state['output'] = function()
        return PreparedCall(launch, lambda : state['output'])
    return checked_call(side(lambda : runtime.varlen_causal_conv1d(x, offsets, weight, bias)), (Tolerance(atol=0.05, rtol=0.05), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有变长卷积及final-state两kernel完整调用；原长度2048/1536/1024/512、D4096、bf16、chunk64；同算法PyTorch eager CPU reference，单NUMA8核；含ABI、输出分配与同步。', owners=locals())
