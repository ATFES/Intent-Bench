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


@pytest.mark.parametrize("dtype", ["float32", "int64"])
def test_bangc_histogram_all_filter_groups_and_tail(dtype):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_mlu")
    if not os.getenv("INTENTDSL_LIBRARY") or not torch.mlu.is_available():
        pytest.skip("requires an installed IntentDSL product and MLU")
    import intentdsl
    from intentdsl import examples

    values = [0, 1, 63, 64, 127, 128, 191, 192, 255, 256, -7]
    if dtype == "int64":
        values += [65536, 1 << 32, (1 << 32) + 1, -(1 << 32), 1 << 62]
        program, extra = examples.histogram_from_indices, (256,)
    else:
        values += [-0.75, 1.75, 65536., -65536.]
        program, extra = examples.histogram, ()
    n = 16 * 8192 + 37
    host = torch.tensor(values, dtype=getattr(torch, dtype)).repeat(n // len(values) + 1)[:n]
    keys = host.long()
    expected = torch.bincount(keys[(keys >= 0) & (keys < 256)], minlength=256)
    actual = intentdsl.jit(program, backend="bangc").run(host.to("mlu"), *extra).cpu()
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)
