"""Complete native MAX Q/K RoPE on the installed author's original buffers."""
from pathlib import Path
from types import SimpleNamespace
import ctypes as C

from ..model import PreparedCall
from .native_sampling import sample_timer


def prepare(arguments, environment):
    import torch
    from ..corpus.cpu.baselines.mojo.source import prepare_source, scalar, view
    query, key, cosine, sine = arguments
    if environment.device != "cpu":
        raise ValueError("MAX CPU RoPE requires device=cpu")
    if (query.ndim != 4 or key.ndim != 4 or query.shape[1] != 32 or key.shape[1] != 8
            or query.shape[-1] != 128 or key.shape[-1] != 128
            or query.shape[0] != key.shape[0] or query.shape[2] != key.shape[2]
            or cosine.shape != (1, query.shape[2], 128) or sine.shape != cosine.shape):
        raise ValueError("MAX binding requires the authored split-half Q[B,32,S,128]/K[B,8,S,128] contract")
    if any(value.dtype != torch.float16 for value in arguments):
        raise ValueError("MAX RoPE binding preserves the author's FP16 operands")
    # B/H/S/D storage is already contiguous across independent head sequences.
    # These row boundaries depend only on bound shapes, never input contents.
    sequence = query.shape[2]
    counts = (query.shape[0] * query.shape[1], key.shape[0] * key.shape[1])
    if sequence <= 0 or max(counts) * sequence > 2**31 - 1:
        raise ValueError("MAX RoPE sequence boundaries must fit positive signed i32")
    indices = tuple(value for count in counts for value in (
        torch.arange(count + 1, dtype=torch.int32) * sequence,
        torch.zeros(count, dtype=torch.int32)))
    source = Path(__file__).parents[1]/"native/max_rope.mojo"
    context = C.c_void_p()
    native = prepare_source(source, "max_rope_qk", [
        view("query", (1, 2, 3, 4), dtype="f16", compact=True),
        view("key", (1, 5, 3, 4), dtype="f16", compact=True),
        view("cosine", (6, 3, 4), dtype="f16", compact=True),
        view("sine", (6, 3, 4), dtype="f16", compact=True),
        view("frequencies", (3, 4), dtype="f16", compact=True, output=True),
        view("query_offsets", (7,), dtype="i32", compact=True),
        view("query_positions", (8,), dtype="i32", compact=True),
        view("key_offsets", (9,), dtype="i32", compact=True),
        view("key_positions", (10,), dtype="i32", compact=True),
        scalar("query_sequences", "i64"), scalar("key_sequences", "i64"), scalar("sequence", "i64"),
        scalar("context", "ptr"),
    ], SimpleNamespace(workers=environment.threads, executable=environment.options.get("mojo")),
        (*arguments, *indices, *counts, sequence, context))
    size = native.library.max_rope_context_size
    alignment = native.library.max_rope_context_alignment
    for function in (size, alignment):
        function.argtypes = []
        function.restype = C.c_int64
    initialize = native.library.max_rope_context_initialize
    destroy = native.library.max_rope_context_destroy
    for function in (initialize, destroy):
        function.argtypes = [C.c_void_p]
        function.restype = None
    align = alignment()
    storage = C.create_string_buffer(size() + align - 1)
    context.value = (C.addressof(storage) + align - 1) // align * align
    initialize(context)
    def close():
        if context.value:
            destroy(context)
            context.value = None
    metadata = {"library": "MAX native split-half RoPE",
                "reference_source": "nn.rope.rope_ragged",
                "binding_source": str(source),
                "scope_note": "Both Q and K, phase interleaving, native dispatch and synchronization are timed; context construction and shape-only row bindings are prepared once. Original InOut buffers retained."}
    call = PreparedCall(native.launch, lambda: (query, key), device="cpu",
        timing="cpu_native_monotonic", owners=(native, arguments, storage), close=close, metadata=metadata)
    call.native_measure = lambda settings: sample_timer(native, settings,
        call.reset or (lambda: None), metadata=metadata)
    return call
