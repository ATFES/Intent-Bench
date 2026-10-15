from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.streaming.paged_attention import splitk_paged_gqa_decode_partials
from kernels.streaming.splitk_reduce import splitk_attention_bf16_to_f16_reduce
from intent_bench.corpus._common.loading import load_module
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance

def prepare(context):
    _inlined_paged_gqa_split_q = torch.randn((16, 32, 128), dtype=torch.float16)
    _inlined_paged_gqa_split_key = torch.randn((8192, 16, 8, 128), dtype=torch.float16)
    _inlined_paged_gqa_split_value = torch.randn_like(_inlined_paged_gqa_split_key)
    _inlined_paged_gqa_split_indices = torch.arange(8192, dtype=torch.int32)
    _inlined_paged_gqa_split_offsets = torch.arange(0, 8193, 512, dtype=torch.int32)
    _inlined_paged_gqa_split_splits = torch.arange(0, 8193, 64, dtype=torch.int32)
    _inlined_paged_gqa_split_lengths = torch.full((16,), 8192, dtype=torch.int32)
    _inlined_paged_gqa_split_arguments = (_inlined_paged_gqa_split_q, _inlined_paged_gqa_split_key, _inlined_paged_gqa_split_value, _inlined_paged_gqa_split_offsets, _inlined_paged_gqa_split_indices, _inlined_paged_gqa_split_lengths, _inlined_paged_gqa_split_splits, 128 ** (-0.5))
    _inlined_paged_gqa_split_partials = intent.compile(splitk_paged_gqa_decode_partials, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options, constexprs={'PAGE_SIZE': 16, 'HEAD_GROUP': 4, 'SPLITS': 8})
    _inlined_paged_gqa_split_reduction = intent.compile(splitk_attention_bf16_to_f16_reduce, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options)
    (_inlined_paged_gqa_split_generated, _inlined_paged_gqa_split_source) = ({}, {})

    def _inlined_paged_gqa_split_launch():
        (_inlined_paged_gqa_split_lse, _inlined_paged_gqa_split_partial) = _inlined_paged_gqa_split_partials.run(*_inlined_paged_gqa_split_arguments)
        _inlined_paged_gqa_split_generated['output'] = _inlined_paged_gqa_split_reduction.run(_inlined_paged_gqa_split_partial, _inlined_paged_gqa_split_lse)
    if False:
        _inlined_paged_gqa_split_runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

        def _inlined_paged_gqa_split_reference():
            _inlined_paged_gqa_split_source['output'] = _inlined_paged_gqa_split_runtime.paged_gqa_decode_bf16(*_inlined_paged_gqa_split_arguments)
        _inlined_paged_gqa_split_source_launch = PreparedLaunch(_inlined_paged_gqa_split_reference, lambda : _inlined_paged_gqa_split_source['output'])
        (_inlined_paged_gqa_split_tolerance, _inlined_paged_gqa_split_status) = (Tolerance(atol=0.1, rtol=0.05), 'pass')
        _inlined_paged_gqa_split_note = '原B16/HQ32/HK8/K8192/D128/page16/split8 f16输入，BF16 partial+weighted BF16 reduce完整调用含workspace分配；CPU参考page-table attention f32概率后直接转BF16，原容差。'
    else:
        _inlined_paged_gqa_split_runtime = load_module(context.project_root / 'corpus/cpu/baselines/pytorch/cpu_runtime.py', 'intent_cpu_reference')

        def _inlined_paged_gqa_split_reference():
            _inlined_paged_gqa_split_source['output'] = _inlined_paged_gqa_split_runtime._paged_gqa_decode(_inlined_paged_gqa_split_q, _inlined_paged_gqa_split_key, _inlined_paged_gqa_split_value, _inlined_paged_gqa_split_offsets, _inlined_paged_gqa_split_indices, _inlined_paged_gqa_split_lengths, 128 ** (-0.5)).half()
        _inlined_paged_gqa_split_source_launch = PreparedLaunch(_inlined_paged_gqa_split_reference, lambda : _inlined_paged_gqa_split_source['output'])
        (_inlined_paged_gqa_split_tolerance, _inlined_paged_gqa_split_status) = (Tolerance(atol=0.1, rtol=0.05), 'pass')
        _inlined_paged_gqa_split_note = 'BF16 partials with f16 final reduction; complete page-table attention checked against independent f32 CPU reference, preserving the existing BF16-partial tolerance.'
    return Side(PreparedLaunch(_inlined_paged_gqa_split_launch, lambda : _inlined_paged_gqa_split_generated['output']), _inlined_paged_gqa_split_tolerance, cuda_graph=False, status=_inlined_paged_gqa_split_status, device_type='cpu', note=_inlined_paged_gqa_split_note, owners=locals())
