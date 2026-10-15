from __future__ import annotations
from intent_bench.sides import Side
import intent
import torch
from kernels.variants.decomposition import nested_sentence_pool, nested_document_pool
from intent_bench.corpus._common.model import PreparedLaunch, Tolerance
DOCUMENTS = 256
NESTED_FEATURES = 128
SENTENCES = 4096
NESTED_TOKENS = 65536

def prepare(context):
    (document_offsets, sentence_offsets, values) = _nested_pool_inputs()
    (sentences, documents) = (intent.compile(definition, target=context.target, compiler=context.compiler, tuning_config=context.tuning_config, options=context.compile_options) for definition in (nested_sentence_pool, nested_document_pool))
    (generated, source) = ({}, {})

    def launch():
        (sums, means) = sentences.run(sentence_offsets, values)
        generated['output'] = (means, documents.run(document_offsets, sentence_offsets, sums))
    return Side(PreparedLaunch(launch, lambda : generated['output']), (Tolerance(atol=2e-05), Tolerance(atol=2e-05)), cuda_graph=False, device_type='cpu', note='原D256/S4096/T65536/F128 f32 nested pool拆分变体；8/24交替正长度，sentence sum/mean后document mean，完整两kernel调用含scratch和输出分配；CPU fused数学参考、原容差。', owners=locals())

def _nested_pool_inputs():
    document_lengths = torch.tensor([8 if index % 2 == 0 else 24 for index in range(DOCUMENTS)], dtype=torch.int32)
    sentence_lengths = torch.tensor([8 if index % 2 == 0 else 24 for index in range(SENTENCES)], dtype=torch.int32)
    document_offsets = torch.zeros(DOCUMENTS + 1, dtype=torch.int32)
    sentence_offsets = torch.zeros(SENTENCES + 1, dtype=torch.int32)
    document_offsets[1:] = document_lengths.cumsum(dim=0)
    sentence_offsets[1:] = sentence_lengths.cumsum(dim=0)
    values = torch.randn((NESTED_TOKENS, NESTED_FEATURES), dtype=torch.float32)
    return (document_offsets, sentence_offsets, values)
