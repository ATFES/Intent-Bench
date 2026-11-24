"""Minimal regression tests for audit-mig GPU correctness and legalization rules."""

import json
import os

import pytest


def _require_library():
    library = os.environ.get("INTENTDSL_LIBRARY")
    if not library:
        pytest.skip("requires INTENTDSL_LIBRARY pointing to libintentdsl.so")
    return library


@pytest.mark.parametrize("exit_name", ["triton", "cutile"])
def test_owner_rectangle_mismatched_domain_and_view_tail(exit_name):
    """Owner rectangle proof must check actual owner-domain start offsets, not view extent - tile."""
    library = _require_library()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA device")
    if exit_name == "cutile":
        pytest.importorskip("cuda.tile")
    import intentdsl as I

    @I.kernel
    def paired_straddling_domain(
        q: I.In[I.bf16, ("B", "M_PAD", "D")],
        k: I.In[I.bf16, ("B", "N", "D")],
        y: I.Out[I.f32, ("B", "M", "D")],
    ):
        for b in I.parallel(y.dim(0)):
            for m in I.parallel(y.dim(1)):
                q_row = I.cast(q[b, m, :], I.f32)
                acc = I.zeros(y.dim(2), I.f32)
                for t in I.fold(k.dim(1)):
                    kv = I.cast(k[b, t, :], I.f32)
                    acc = acc + kv * I.sum(q_row * kv)
                y[b, m, :] = acc

    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(7)

    # Case 1: y has M=50 (not divisible by owner_tile in {16,32,64}), while q has M_PAD=128.
    # Poison q[:, 50:, :] so straddling CTAs reading q[0, 50..63, :] for batch 1 fail immediately.
    q = torch.full((2, 128, 64), 1024.0, device=device, dtype=torch.bfloat16)
    q[:, :50, :] = (torch.randn((2, 50, 64), generator=gen, device=device) * 0.25).to(torch.bfloat16)
    k = (torch.randn((2, 64, 64), generator=gen, device=device) * 0.25).to(torch.bfloat16)
    y = torch.empty((2, 50, 64), device=device, dtype=torch.float32)

    fn = I.jit(paired_straddling_domain, backend=exit_name, library=library)
    root = fn.compiler.root_ir(paired_straddling_domain.source)
    with fn.compilation_target(q, k, outputs=(y,)) as target:
        legal_ir = target.compile(root, exit=exit_name, through=f"{exit_name}@legal", text=True).decode()
    assert "owner_rectangle" not in legal_ir

    fn(q, k, y)
    torch.cuda.synchronize(device)
    qf = q[:, :50, :].float()
    kf = k.float()
    expected = torch.einsum("bmn,bnd->bmd", torch.einsum("bmd,bnd->bmn", qf, kf), kf)
    torch.testing.assert_close(y, expected, rtol=1e-2, atol=1e-2)

    # Case 2: y has M=192 (multiple of 64) while q only has M_PAD=128 (< 192).
    # Actual last tile start on q is 192 - tile >= 128, so q's slice_read must not get owner_rectangle.
    q_short = torch.zeros((1, 128, 64), device=device, dtype=torch.bfloat16)
    k_one = torch.zeros((1, 64, 64), device=device, dtype=torch.bfloat16)
    y_long = torch.empty((1, 192, 64), device=device, dtype=torch.float32)
    with fn.compilation_target(q_short, k_one, outputs=(y_long,)) as target:
        legal_ir_oob = target.compile(root, exit=exit_name, through=f"{exit_name}@legal", text=True).decode()
    for line in legal_ir_oob.splitlines():
        if "slice_read" in line:
            assert "owner_rectangle" not in line


def test_cutile_nsz_preserves_subnormals_under_ieee_denormal_mode():
    """nsz must never trigger flush_to_zero=True; only function denormal-fp-math controls FTZ."""
    library = _require_library()
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuda.tile")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA device")
    import intentdsl as I

    @I.kernel
    def row_subnormal_div(
        x: I.In[I.f32, ("M", "N")],
        y: I.In[I.f32, ("M", "N")],
        out: I.Out[I.f32, ("M", "N")],
    ):
        for m in I.parallel(x.dim(0)):
            out[m, :] = x[m, :] / y[m, :]

    @I.kernel
    def ordered_subnormal_div(
        x: I.In[I.f32, ("M", "N")],
        y: I.In[I.f32, ("M", "N")],
        out: I.Out[I.f32, ("M", "N")],
    ):
        for m in I.parallel(x.dim(0)):
            acc = I.zeros(x.dim(1), I.f32)
            for _ in I.ordered(1):
                acc = acc + x[m, :] / y[m, :]
            out[m, :] = acc

    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.full((4, 32), 1.4e-40, device=device, dtype=torch.float32)
    y = torch.full((4, 32), 2.0, device=device, dtype=torch.float32)
    expected = x / y
    assert (expected != 0).all()

    for kernel_def in (row_subnormal_div, ordered_subnormal_div):
        fn_ieee = I.jit(
            kernel_def,
            backend="cutile",
            library=library,
            fast_math=True,
            denormal_fp_math="ieee",
        )
        root = fn_ieee.compiler.root_ir(kernel_def.source)
        out_ieee = torch.empty_like(x)
        with fn_ieee.compilation_target(x, y, outputs=(out_ieee,)) as target:
            emitted, _ = target.compile_runtime(root, exit="cutile")
        src_ieee = json.loads(emitted)["source"]
        assert "flush_to_zero=True" not in src_ieee

        fn_ieee(x, y, out_ieee)
        torch.cuda.synchronize(device)
        assert (out_ieee != 0).all()
        assert torch.equal(out_ieee.view(torch.int32), expected.view(torch.int32))

        fn_ftz = I.jit(
            kernel_def,
            backend="cutile",
            library=library,
            fast_math=True,
            denormal_fp_math="preserve-sign",
        )
        out_ftz = torch.empty_like(x)
        with fn_ftz.compilation_target(x, y, outputs=(out_ftz,)) as target:
            emitted_ftz, _ = target.compile_runtime(root, exit="cutile")
        src_ftz = json.loads(emitted_ftz)["source"]
        assert "flush_to_zero=True" in src_ftz

        fn_ftz(x, y, out_ftz)
        torch.cuda.synchronize(device)
        assert (out_ftz == 0).all()


def test_cutile_scf_for_loop_kind_decided_in_legalize():
    """cuTile scf.for range vs while lowering choice must be materialized and verified in IR."""
    library = _require_library()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA device")
    import intentdsl as I

    @I.kernel
    def static_loop_kernel(
        x: I.In[I.f32, ("M", "N")],
        out: I.Out[I.f32, ("M", "N")],
    ):
        for m in I.parallel(x.dim(0)):
            acc = I.zeros(x.dim(1), I.f32)
            for _ in I.ordered(3):
                acc = acc + x[m, :]
            out[m, :] = acc

    @I.kernel
    def dynamic_loop_kernel(
        x: I.In[I.f32, ("M", "K")],
        lengths: I.In[I.i32, ("M",)],
        out: I.Out[I.f32, ("M",)],
    ):
        for m in I.parallel(x.dim(0)):
            n = I.cast(lengths[m], I.index)
            acc = I.f32(0)
            for k in I.ordered(n):
                acc = acc + x[m, k]
            out[m] = acc

    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.ones((8, 32), device=device, dtype=torch.float32)
    out2d = torch.empty_like(x)
    lengths = torch.full((8,), 16, device=device, dtype=torch.int32)
    out1d = torch.empty((8,), device=device, dtype=torch.float32)

    fn_static = I.jit(static_loop_kernel, backend="cutile", library=library)
    with fn_static.compilation_target(x, outputs=(out2d,)) as target:
        ir_static = target.compile(
            fn_static.compiler.root_ir(static_loop_kernel.source),
            exit="cutile",
            through="cutile@callable",
            text=True,
        ).decode()
    assert 'ct_src.loop_kind = "range_const"' in ir_static
    assert "ct_src.range_bounds = [0, 3, 1]" in ir_static

    fn_dyn = I.jit(dynamic_loop_kernel, backend="cutile", library=library)
    with fn_dyn.compilation_target(x, lengths, outputs=(out1d,)) as target:
        ir_dyn = target.compile(
            fn_dyn.compiler.root_ir(dynamic_loop_kernel.source),
            exit="cutile",
            through="cutile@callable",
            text=True,
        ).decode()
        assert 'ct_src.loop_kind = "while"' in ir_dyn

        tampered = ir_dyn.replace('ct_src.loop_kind = "while"', 'ct_src.loop_kind = "range_const"')
        with pytest.raises(Exception, match="scf.for lowering decision"):
            target.compile(tampered.encode(), exit="cutile", after="cutile@callable")
