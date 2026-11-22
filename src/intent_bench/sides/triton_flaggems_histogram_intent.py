from __future__ import annotations
from intent_bench.implementations.prepared import checked_call
from intent.examples.kernels.statistics.histogram import histogram_partials, histogram_merge
from intent.examples.programs.composition import Histogram
from intent_bench.corpus._common.measurement import compile_kernel, prepare_program
from intent_bench.implementations.context import Tolerance

def prepare(context):
    samples, = context.inputs
    chunk = 8192
    program = Histogram(compile_kernel(context, histogram_partials, constexprs={"CHUNK": chunk}),
                        compile_kernel(context, histogram_merge), chunk)
    generated = prepare_program(program, (samples,))
    context.prepared_calls.append(generated)
    return checked_call(generated, Tolerance(atol=0.0), cuda_graph=False, owners=locals())
