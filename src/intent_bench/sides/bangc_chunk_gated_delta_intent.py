from intent.examples.kernels.streaming.gated_delta import chunk_gated_delta_prepare, chunk_gated_delta_recurrence
from intent.examples.programs.composition import ChunkGatedDelta
from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.prepared import checked_call
from intent_bench.implementations.bangc import prepare_program


def prepare(context):
    program = ChunkGatedDelta(context.compile(chunk_gated_delta_prepare),
                              context.compile(chunk_gated_delta_recurrence))
    tolerance = Tolerance(atol=0.1, rtol=0.05)
    return checked_call(prepare_program(context, program), (tolerance, tolerance),
                        device_type='mlu', owners={'program': program})
