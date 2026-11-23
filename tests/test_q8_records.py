import ctypes
import ctypes.util
import os

import numpy as np
import pytest

from intent_bench.references.quantization import q8_records


def test_q8_bytes_against_ggml():
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
    np.testing.assert_array_equal(q8_records(x), reference)
