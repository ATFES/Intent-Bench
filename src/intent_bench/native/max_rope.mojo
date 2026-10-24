# Native binding to MAX's split-half RoPE, preserving the author's Q/K layout.
from layout import Coord, Idx, TileTensor, row_major
from nn.rope import rope_ragged
from max.gpu.host import DeviceContext
from std.os import abort
from std.runtime import initialize_runtime
from std.sys import align_of, size_of
from std.utils import IndexList


@export("max_rope_context_size")
def context_size() abi("C") -> Int64:
    return Int64(size_of[DeviceContext]())


@export("max_rope_context_alignment")
def context_alignment() abi("C") -> Int64:
    return Int64(align_of[DeviceContext]())


@export("max_rope_context_initialize")
def context_initialize(context: Pointer[DeviceContext, MutUntrackedOrigin]) abi("C"):
    initialize_runtime()
    try:
        context.unsafe_write(copy=DeviceContext(api="cpu"))
    except error:
        abort(String("MAX RoPE context failed: ", error))


@export("max_rope_context_destroy")
def context_destroy(context: Pointer[DeviceContext, MutUntrackedOrigin]) abi("C"):
    context.unsafe_deinit_pointee()


def rotate(
    values: Pointer[Float16, MutUntrackedOrigin],
    frequencies: Pointer[Float16, MutUntrackedOrigin],
    offsets: Pointer[Int32, MutUntrackedOrigin],
    positions: Pointer[Int32, MutUntrackedOrigin],
    sequences: Int,
    sequence: Int,
    context: DeviceContext,
) raises:
    var x = TileTensor(values, row_major(Coord(sequences * sequence, Idx[1], Idx[128])))
    var freq = TileTensor(frequencies, row_major(Coord(sequence, Idx[128])))
    var rows = TileTensor(offsets.unsafe_bitcast[UInt32](), row_major(sequences + 1))
    var starts = TileTensor(positions.unsafe_bitcast[UInt32](), row_major(sequences))

    @always_inline
    def write[width: SIMDLength, alignment: Int](
        index: IndexList[3], value: SIMD[DType.float16, width]
    ) {var x}:
        x.store[width=width](Coord(index), value)

    rope_ragged[DType.float16, DType.float16, interleaved=False, target="cpu"](
        x, rows, starts, freq, context, write
    )


@export("max_rope_qk")
@no_inline
def max_rope_qk(
    query: Pointer[Float16, MutUntrackedOrigin],
    key: Pointer[Float16, MutUntrackedOrigin],
    cosine: Pointer[Float16, MutUntrackedOrigin],
    sine: Pointer[Float16, MutUntrackedOrigin],
    frequencies: Pointer[Float16, MutUntrackedOrigin],
    query_offsets: Pointer[Int32, MutUntrackedOrigin],
    query_positions: Pointer[Int32, MutUntrackedOrigin],
    key_offsets: Pointer[Int32, MutUntrackedOrigin],
    key_positions: Pointer[Int32, MutUntrackedOrigin],
    query_sequences: Int64,
    key_sequences: Int64,
    sequence: Int64,
    context: Pointer[DeviceContext, MutUntrackedOrigin],
) abi("C"):
    # The phase tables remain invocation inputs, so their interleaving is timed.
    for token in range(Int(sequence)):
        for column in range(0, 64, 16):
            var c = cosine.unsafe_offset(token * 128 + column).unsafe_load[width=16]()
            var s = sine.unsafe_offset(token * 128 + column).unsafe_load[width=16]()
            frequencies.unsafe_offset(token * 128 + column * 2).unsafe_store(c.interleave(s))
    try:
        rotate(query, frequencies, query_offsets, query_positions, Int(query_sequences), Int(sequence), context[])
        rotate(key, frequencies, key_offsets, key_positions, Int(key_sequences), Int(sequence), context[])
        context[].synchronize()
    except error:
        abort(String("MAX RoPE failed: ", error))
