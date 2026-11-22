from intent_bench.implementations.context import Tolerance
from intent_bench.implementations.cub import histogram
from intent_bench.implementations.prepared import checked_call


def prepare(context):
    samples, = context.inputs
    call = histogram(samples, context.environment)
    return checked_call(call, Tolerance(atol=0.0), owners={"call_buffers": call.owners})
