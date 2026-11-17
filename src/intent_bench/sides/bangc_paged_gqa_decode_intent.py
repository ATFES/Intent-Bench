from intent.examples.kernels.streaming.paged_attention import paged_gqa_decode_partials
from intent.examples.kernels.streaming.splitk_reduce import splitk_attention_f32_to_f16_reduce
from intent.examples.programs.composition import PagedDecode
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import checked_call
from intent_bench.implementations.bangc import prepare_program


def prepare(context):
    program = PagedDecode(context.compile(paged_gqa_decode_partials,
                         constexprs={'PAGE_SIZE': 16, 'HEAD_GROUP': 4, 'SPLITS': 8}),
                          context.compile(splitk_attention_f32_to_f16_reduce))
    return checked_call(prepare_program(context, program), Tolerance(atol=0.05, rtol=0.05),
                        device_type='mlu', owners={'program': program})
