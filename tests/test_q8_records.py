import ctypes
import ctypes.util
import json
import os
import subprocess

import numpy as np
import pytest

from intent_bench.references.quantization import q8_records


@pytest.fixture
def ggml_blocks():
    library = os.getenv("GGML_BASE_LIBRARY") or ctypes.util.find_library("ggml-base")
    if not library:
        pytest.skip("requires GGML's quantize_row_q8_K_ref")
    encode = ctypes.CDLL(library).quantize_row_q8_K_ref
    encode.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)
    encode.restype = None
    x = np.random.default_rng(713).normal(size=(20, 256)).astype("float32")
    x[0] = -0.
    for i, first in enumerate((127., -127.), 1):
        x[i, :6] = [first, -first, .5, -.5, 1.5, -1.5]
    # GGML leaves bsums untouched in an all-zero record.
    reference = np.zeros((len(x), 292), np.uint8)
    encode(x.ctypes.data, reference.ctypes.data, x.size)
    return x, reference


def test_q8_bytes_against_ggml(ggml_blocks):
    x, reference = ggml_blocks
    np.testing.assert_array_equal(q8_records(x), reference)


def test_packed_host_argument_matches_ggml(ggml_blocks, tmp_path):
    if not os.getenv("INTENTDSL_LIBRARY"):
        pytest.skip("requires an installed CPU product")
    import intentdsl
    from intentdsl import examples

    x, reference = ggml_blocks
    records = np.zeros((1, len(x), 144), np.uint8)
    kernel = intentdsl.jit(examples.q4_projection, backend="mojo")
    with kernel.compilation_target(records, x) as compiler:
        emitted, _ = compiler.compile_runtime(compiler.root_ir(kernel.source), exit="mojo")
    # Observe the physical argument at the host/task ABI. The floating dot
    # result alone cannot distinguish two records with opposite scale and codes.
    host = tmp_path / "entry.c"
    host.write_text(json.loads(emitted)["host"] +
                    f"\nvoid intent_panel(void *w, void *packed, void *out, int64_t m, int64_t n)"
                    f" {{ memcpy(out, packed, {reference.nbytes}); }}\n")
    library = tmp_path / "entry.so"
    subprocess.run(["cc", "-O3", "-fopenmp", "-shared", "-fPIC", host, "-o", library], check=True)
    entry = ctypes.CDLL(str(library)).intent_entry
    ptr, idx = ctypes.c_void_p, ctypes.c_int64
    entry.argtypes = [ptr, *([idx] * 5), ptr, idx, idx, ptr, *([idx] * 6)]
    entry.restype = ctypes.c_int
    actual = np.empty_like(reference)
    assert entry(records.ctypes.data, 1, len(x), len(x)*144, 144, 1,
                 x.ctypes.data, 256, 1, actual.ctypes.data, 1, 1, 1, 1, 1, 1) == 0
    np.testing.assert_array_equal(actual, reference)
