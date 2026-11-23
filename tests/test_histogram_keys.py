"""Regression for an author cast crossing a fused GPU histogram."""
import os

import pytest


@pytest.mark.parametrize("backend", ["triton", "cutile"])
def test_histogram_key_conversion(backend):
    torch = pytest.importorskip("torch")
    if not os.getenv("INTENTDSL_LIBRARY") or not torch.cuda.is_available():
        pytest.skip("requires an installed IntentDSL product and CUDA")
    import intentdsl
    from intentdsl import examples

    # A negative fraction truncates into bin zero. A wide integer that
    # aliases zero in its low 32 bits remains outside the range.
    for program, keys, extra, dtype in (
        (examples.histogram, [-0.75, 1.5] + [-1.25] * 254, (), torch.float32),
        (examples.histogram_from_indices, [0, 1, 1 << 32], (256,), torch.int64),
    ):
        x = torch.tensor(keys, dtype=dtype, device="cuda")
        counts = intentdsl.jit(program, backend=backend).run(x, *extra)
        expected = torch.zeros_like(counts)
        expected[:2] = 1
        torch.testing.assert_close(counts, expected, rtol=0, atol=0)
