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
                q_row = q[b, m, :]
                acc = I.zeros(y.dim(2), I.f32)
                for t in I.fold(k.dim(1)):
                    kv = k[b, t, :]
                    score = I.sum(I.cast(q_row, I.f32) * I.cast(kv, I.f32))
                    p = I.cast(score, I.bf16)
                    acc = acc + I.cast(kv, I.f32) * I.cast(p, I.f32)
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

    # Case 2: y has M=192 (multiple of 64 -> pair_axis selected) while q only has M_PAD=128 (< 192).
    # Actual last tile start on q is 192 - tile >= 128, so q's slice_read must not get owner_rectangle,
    # whereas y's slice_write (extent 192) does get owner_rectangle.
    q_short = torch.zeros((16, 128, 64), device=device, dtype=torch.bfloat16)
    k_one = torch.zeros((16, 64, 64), device=device, dtype=torch.bfloat16)
    y_long = torch.empty((16, 192, 64), device=device, dtype=torch.float32)
    with fn.compilation_target(q_short, k_one, outputs=(y_long,)) as target:
        legal_ir_oob = target.compile(root, exit=exit_name, through=f"{exit_name}@legal", text=True).decode()
    assert "block.pair_axis" in legal_ir_oob
    for line in legal_ir_oob.splitlines():
        if "slice_read" in line and "%arg0" in line:
            assert "owner_rectangle" not in line
    assert any("slice_write" in line and "owner_rectangle" in line for line in legal_ir_oob.splitlines())

    # Case 3 (V05Tests 2B scenario: extent=100 vs extent=128):
    # extent=100 must never get owner_rectangle, while aligned extent=128 does.
    q_100 = (torch.randn((16, 100, 64), generator=gen, device=device) * 0.25).to(torch.bfloat16)
    y_100 = torch.empty((16, 100, 64), device=device, dtype=torch.float32)
    with fn.compilation_target(q_100, k_one, outputs=(y_100,)) as target:
        legal_ir_100 = target.compile(root, exit=exit_name, through=f"{exit_name}@legal", text=True).decode()
    assert "owner_rectangle" not in legal_ir_100
    fn(q_100, k_one, y_100)
    torch.cuda.synchronize(device)
    assert torch.all(y_100 == 0)

    # Case 4 (V05Tests 2A scenario: K=4097 vs K=4096 fold panels, plus extent=128 owner_rectangle):
    # K=4097 has a partial tail panel [4096, 4097) and must not be marked block.full_panels,
    # whereas K=4096 is exact and gets block.full_panels; aligned M=128 gets owner_rectangle.
    q_aligned = (torch.randn((16, 128, 64), generator=gen, device=device) * 0.25).to(torch.bfloat16)
    y_aligned = torch.empty((16, 128, 64), device=device, dtype=torch.float32)
    k_4097 = (torch.randn((16, 4097, 64), generator=gen, device=device) * 0.05).to(torch.bfloat16)
    k_4096 = k_4097[:, :4096, :].contiguous()
    with fn.compilation_target(q_aligned, k_4097, outputs=(y_aligned,)) as target:
        realized_4097 = target.compile(root, exit=exit_name, through="block@realized", text=True).decode()
        legal_4097 = target.compile(root, exit=exit_name, through=f"{exit_name}@legal", text=True).decode()
    assert "block.pair_axis" in realized_4097
    assert "block.full_panels" not in realized_4097
    assert "owner_rectangle" in legal_4097
    with fn.compilation_target(q_aligned, k_4096, outputs=(y_aligned,)) as target:
        realized_4096 = target.compile(root, exit=exit_name, through="block@realized", text=True).decode()
    assert "block.pair_axis" in realized_4096
    assert "block.full_panels" in realized_4096
    fn(q_aligned, k_4097, y_aligned)
    torch.cuda.synchronize(device)
    expected_4097 = torch.einsum(
        "bmn,bnd->bmd",
        torch.einsum("bmd,bnd->bmn", q_aligned.float(), k_4097.float()),
        k_4097.float(),
    )
    torch.testing.assert_close(y_aligned, expected_4097, rtol=2e-2, atol=2e-2)


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


def test_author_flush_to_zero_propagates_denormal_mode():
    """Explicit op-level flush_to_zero=True must set function denormal-fp-math='preserve-sign'."""
    library = _require_library()
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuda.tile")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA device")
    import intentdsl as I

    @I.kernel
    def fdiv_ftz(x: I.In[I.f32, ("M", "N")], y: I.In[I.f32, ("M", "N")], out: I.Out[I.f32, ("M", "N")]):
        for m in I.parallel(x.dim(0)):
            out[m, :] = I.fdiv(x[m, :], y[m, :], approximate=True, flush_to_zero=True)

    @I.kernel
    def fdiv_ieee(x: I.In[I.f32, ("M", "N")], y: I.In[I.f32, ("M", "N")], out: I.Out[I.f32, ("M", "N")]):
        for m in I.parallel(x.dim(0)):
            out[m, :] = I.fdiv(x[m, :], y[m, :], approximate=True)

    @I.kernel
    def ordered_exp2_ftz(x: I.In[I.f32, ("M", "N")], out: I.Out[I.f32, ("M", "N")]):
        for m in I.parallel(x.dim(0)):
            acc = I.zeros(x.dim(1), I.f32)
            for _ in I.ordered(1):
                acc = acc + x[m, :] * I.exp2(I.sum(x[m, :]), approximate=True, flush_to_zero=True)
            out[m, :] = acc

    @I.kernel
    def ordered_exp2_ieee(x: I.In[I.f32, ("M", "N")], out: I.Out[I.f32, ("M", "N")]):
        for m in I.parallel(x.dim(0)):
            acc = I.zeros(x.dim(1), I.f32)
            for _ in I.ordered(1):
                acc = acc + x[m, :] * I.exp2(I.sum(x[m, :]), approximate=True)
            out[m, :] = acc

    device = torch.device("cuda", torch.cuda.current_device())
    x = torch.full((4, 32), 1.4e-40, device=device, dtype=torch.float32)
    y = torch.full((4, 32), 2.0, device=device, dtype=torch.float32)
    out = torch.empty_like(x)

    for k_ftz, k_ieee, ins in (
        (fdiv_ftz, fdiv_ieee, (x, y)),
        (ordered_exp2_ftz, ordered_exp2_ieee, (x,)),
    ):
        fn_ftz = I.jit(k_ftz, backend="cutile", library=library)
        root_ftz = fn_ftz.compiler.root_ir(k_ftz.source)
        assert '"preserve-sign"' in root_ftz.decode()
        with fn_ftz.compilation_target(*ins, outputs=(out,)) as target:
            emitted_ftz, _ = target.compile_runtime(root_ftz, exit="cutile")
        assert "flush_to_zero=True" in json.loads(emitted_ftz)["source"]

        fn_ieee = I.jit(k_ieee, backend="cutile", library=library)
        root_ieee = fn_ieee.compiler.root_ir(k_ieee.source)
        assert '"preserve-sign"' not in root_ieee.decode()
        with fn_ieee.compilation_target(*ins, outputs=(out,)) as target:
            emitted_ieee, _ = target.compile_runtime(root_ieee, exit="cutile")
        assert "flush_to_zero=True" not in json.loads(emitted_ieee)["source"]


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


@pytest.mark.parametrize("exit_name", ["triton", "cutile"])
def test_non_divisible_radix_and_exact_half_bias_awq(exit_name):
    """Regression: non-divisible tail masks in radix_pass/depthwise and FP16 0x6400 AWQ decode."""
    library = _require_library()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA device")
    if exit_name == "cutile":
        pytest.importorskip("cuda.tile")
    import numpy as np
    import intentdsl as I
    from intentdsl.group7_cpu import radix_pass_rows
    from intentdsl.group7_dsa import awq_int4_gemm, depthwise_conv2d

    device = torch.device("cuda", torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(17)

    # 1. Non-divisible radix_pass_rows (B=10, N=100: neither divides P=256/512/64/32)
    keys = torch.randint(0, 100000, (10, 100), generator=gen, device=device, dtype=torch.int32)
    vals = torch.arange(100, device=device, dtype=torch.int32).unsqueeze(0).expand(10, 100).contiguous()
    hist = torch.zeros((10, 256), device=device, dtype=torch.int32)
    cur = torch.zeros((10, 256), device=device, dtype=torch.int32)
    fn_rad = I.jit(radix_pass_rows, backend=exit_name, library=library)
    call_rad = fn_rad.prepare(keys, vals, hist, cur, 256)
    src_rad = json.loads(call_rad.artifact.emitted.decode())["source"]
    assert "mask=" in src_rad
    if exit_name == "triton":
        assert "tl.histogram(" in src_rad and "tl.gather(" in src_rad
    else:
        assert "ct.atomic_xchg(" in src_rad and "ct.atomic_add(" in src_rad
    call_rad.launch()
    torch.cuda.synchronize(device)
    _, _, offsets, out_keys, out_vals = call_rad.result()
    k_np = keys.cpu().numpy()
    v_np = vals.cpu().numpy()
    digits = (k_np // 256) % 256
    order = np.argsort(digits, axis=-1, kind="stable")
    assert torch.equal(out_keys.cpu(), torch.from_numpy(np.take_along_axis(k_np, order, axis=-1)))
    assert torch.equal(out_vals.cpu(), torch.from_numpy(np.take_along_axis(v_np, order, axis=-1)))
    for r in range(10):
        exp_off = np.zeros(257, dtype=np.int32)
        np.cumsum(np.bincount(digits[r], minlength=256), out=exp_off[1:])
        assert torch.equal(offsets[r].cpu(), torch.from_numpy(exp_off))

    # 2. Non-divisible depthwise_conv2d (1*3*7*7=147 owners, not divisible by tile)
    x_dw = torch.randn((1, 3, 7, 7), generator=gen, device=device, dtype=torch.float32)
    w_dw = torch.randn((3, 3, 3), generator=gen, device=device, dtype=torch.float32)
    b_dw = torch.randn((3,), generator=gen, device=device, dtype=torch.float32)
    fn_dw = I.jit(depthwise_conv2d, backend=exit_name, library=library)
    call_dw = fn_dw.prepare(x_dw, w_dw, b_dw)
    assert "mask=" in json.loads(call_dw.artifact.emitted.decode())["source"]
    call_dw.launch()
    torch.cuda.synchronize(device)
    ref_dw = torch.nn.functional.conv2d(x_dw, w_dw.unsqueeze(1), b_dw, padding=1, groups=3)
    torch.testing.assert_close(call_dw.result(), ref_dw, rtol=1e-4, atol=1e-4)

    # 3. Exact FP16 magic-bias (0x6400) decode across all 4-bit codes 0..15
    M, N, K, G = 128, 128, 128, 1
    x_awq = (torch.randn((M, K), generator=gen, device=device) * 0.25).to(torch.float16)
    shifts = [0, 16, 4, 20, 8, 24, 12, 28]
    w_codes = torch.arange(K * N, device=device, dtype=torch.int32).reshape(K, N) % 16
    z_codes = (torch.arange(N, device=device, dtype=torch.int32).reshape(G, N) * 3) % 16
    qw = torch.zeros((K, N // 8), device=device, dtype=torch.int32)
    qz = torch.zeros((G, N // 8), device=device, dtype=torch.int32)
    for j, sh in enumerate(shifts):
        qw |= (w_codes[:, j::8] & 0xF) << sh
        qz |= (z_codes[:, j::8] & 0xF) << sh
    sc = (torch.rand((G, N), generator=gen, device=device) * 0.5 + 0.1).to(torch.float16)
    fn_awq = I.jit(awq_int4_gemm, backend=exit_name, library=library)
    call_awq = fn_awq.prepare(x_awq, qw, sc, qz)
    src_awq = json.loads(call_awq.artifact.emitted.decode())["source"]
    assert "25600" in src_awq
    call_awq.launch()
    torch.cuda.synchronize(device)
    w_ref = ((w_codes - z_codes).to(torch.float16) * sc).to(torch.float32)
    expected_awq = (x_awq.to(torch.float32) @ w_ref).to(torch.float16)
    torch.testing.assert_close(call_awq.result(), expected_awq, rtol=1e-2, atol=2e-2)
