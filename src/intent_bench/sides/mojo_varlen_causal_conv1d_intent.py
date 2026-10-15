from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.convolution.varlen import varlen_aligned_causal_depthwise_conv1d, varlen_causal_conv1d_final_state
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    lengths = (2048, 1536, 1024, 512)
    offsets = torch.tensor((0, 2048, 3584, 4608, 5120), dtype=torch.int64)
    chunks = torch.tensor([(sequence, chunk) for (sequence, length) in enumerate(lengths) for chunk in range((length + 63) // 64)], dtype=torch.int32)
    x = torch.randn((sum(lengths), 4096), dtype=torch.bfloat16)
    weight = torch.randn((4096, 4), dtype=torch.float32)
    bias = torch.randn((4096,), dtype=torch.float32)
    forward = intent.compile(varlen_aligned_causal_depthwise_conv1d, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    final_state = intent.compile(varlen_causal_conv1d_final_state, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)

    def side(function):
        state = {}

        def launch():
            state['output'] = function()
        return PreparedLaunch(launch, lambda : state['output'])
    return Side(side(lambda : (forward.run(x, offsets, chunks, weight, bias), final_state.run(x, offsets))), (Tolerance(atol=0.05, rtol=0.05), Tolerance(atol=0.0)), cuda_graph=False, device_type='cpu', note='既有变长卷积及final-state两kernel完整调用；原长度2048/1536/1024/512、D4096、bf16、chunk64；同算法PyTorch eager CPU reference，单NUMA8核；含ABI、输出分配与同步。', owners=locals())
