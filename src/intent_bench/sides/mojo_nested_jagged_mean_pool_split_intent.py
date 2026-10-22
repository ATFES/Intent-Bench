from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
import intent
import torch
from intent.examples.kernels.variants.decomposition import nested_sentence_pool, nested_document_pool
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import IntentSequence
DOCUMENTS = 256
NESTED_FEATURES = 128
SENTENCES = 4096
NESTED_TOKENS = 65536

def prepare(context):
    (document_offsets, sentence_offsets, values) = _nested_pool_inputs()
    (sentences, documents) = (context.compile(definition) for definition in (nested_sentence_pool, nested_document_pool))
    (generated, source) = ({}, {})

    def launch():
        (sums, means) = sentences.run(sentence_offsets, values)
        generated['output'] = (means, documents.run(document_offsets, sentence_offsets, sums))
    sequence = IntentSequence()
    sums, means = sequence.add(sentences, sentence_offsets, values)
    document_means = sequence.add(documents, document_offsets, sentence_offsets, sums)
    call = sequence.prepared(lambda: (means, document_means), host_run=launch, host_result=lambda: generated['output'])
    return checked_call(call, (Tolerance(atol=2e-05), Tolerance(atol=2e-05)), device_type='cpu', note='Prepared D256/S4096/T65536/F128 f32 nested pool split; original alternating 8/24 lengths and both sentence/document kernels.', owners=locals())

def _nested_pool_inputs():
    document_lengths = torch.tensor([8 if index % 2 == 0 else 24 for index in range(DOCUMENTS)], dtype=torch.int32)
    sentence_lengths = torch.tensor([8 if index % 2 == 0 else 24 for index in range(SENTENCES)], dtype=torch.int32)
    document_offsets = torch.zeros(DOCUMENTS + 1, dtype=torch.int32)
    sentence_offsets = torch.zeros(SENTENCES + 1, dtype=torch.int32)
    document_offsets[1:] = document_lengths.cumsum(dim=0)
    sentence_offsets[1:] = sentence_lengths.cumsum(dim=0)
    values = torch.randn((NESTED_TOKENS, NESTED_FEATURES), dtype=torch.float32)
    return (document_offsets, sentence_offsets, values)
