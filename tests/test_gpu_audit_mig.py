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


@pytest.mark.parametrize("exit_name", ["triton", "cutile"])
def test_full_panels_min_clamp_trans_mma_grouped_panel_and_remat(exit_name):
    """Regression: min-clamp full_panels & SM90 transMma, groupedPanel coalescing, and resident remat."""
    library = _require_library()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA device")
    if exit_name == "cutile":
        pytest.importorskip("cuda.tile")
    import intentdsl as I
    from intentdsl.decode import grouped_partials
    from intentdsl import examples, group7_cpu

    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    sm = major * 10 + minor
    gen = torch.Generator(device=device).manual_seed(29)

    # 1. Min-clamp full_panels proof on divisible T=1024 vs non-divisible T=1000
    B, H, KV, D = 8, 32, 8, 128
    scale = D ** -0.5
    for T, expect_full in [(1024, True), (1000, False)]:
        S = (T + 255) // 256
        q = torch.randn((B, H, 1, D), generator=gen, device=device, dtype=torch.bfloat16)
        k = torch.randn((B, KV, T, D), generator=gen, device=device, dtype=torch.bfloat16)
        v = torch.randn((B, KV, T, D), generator=gen, device=device, dtype=torch.bfloat16)
        logs = torch.empty((B, H, S), device=device, dtype=torch.float32)
        partial = torch.empty((B, H, S, D), device=device, dtype=torch.bfloat16)
        fn_gp = I.jit(grouped_partials, backend=exit_name, library=library)
        with fn_gp.compilation_target(q, k, v, scale, outputs=(logs, partial)) as target:
            ir_gp = target.compile(
                fn_gp.compiler.root_ir(grouped_partials.source),
                exit=exit_name,
                through="block@realized",
                text=True,
            ).decode()
        assert ("block.full_panels" in ir_gp) == expect_full
        call_gp = fn_gp.prepare(q, k, v, scale, outputs=(logs, partial))
        art_gp = json.loads(call_gp.artifact.emitted.decode())
        src_gp = art_gp["source"]
        if expect_full and exit_name == "cutile" and sm == 90:
            # SM90 narrow-M transposed WGMMA emits (span, 16) score accumulator
            assert (
                "ct.full((64,16),0,ct.float32)" in src_gp
                or "ct.full((128,16),0,ct.float32)" in src_gp
            )
        call_gp.launch()
        torch.cuda.synchronize(device)
        # Spot-check batch 0 on first and last split against FP32/BF16 reference
        qf = q[:1].float()
        kf = k[:1].float().repeat_interleave(H // KV, dim=1)
        vf = v[:1].float().repeat_interleave(H // KV, dim=1)
        for s_idx in {0, S - 1}:
            lo, hi = s_idx * 256, min((s_idx + 1) * 256, T)
            sc = torch.einsum("bhqd,bhkd->bhk", qf, kf[:, :, lo:hi, :]) * scale
            p_ref = torch.softmax(sc, dim=-1).to(torch.bfloat16).float()
            out_ref = torch.einsum("bhk,bhkd->bhd", p_ref, vf[:, :, lo:hi, :]).to(torch.bfloat16)
            torch.testing.assert_close(partial[:1, :, s_idx, :], out_ref, rtol=2e-2, atol=2e-2)

    # 2. Grouped panel coalescing on rows=1 (CLA) and rows>1 (retention_recurrence)
    if exit_name == "cutile":
        qc = torch.randn((2, 64, 32), generator=gen, device=device, dtype=torch.float32)
        kc = torch.randn((2, 64, 32), generator=gen, device=device, dtype=torch.float32)
        vc = torch.randn((2, 64, 64), generator=gen, device=device, dtype=torch.float32)
        oc = torch.empty_like(vc)
        fc = torch.empty((2, 32, 64), device=device, dtype=torch.float32)
        fn_cla = I.jit(examples.causal_linear_attention, backend="cutile", library=library)
        call_cla = fn_cla.prepare(qc, kc, vc, outputs=(oc, fc))
        src_cla = json.loads(call_cla.artifact.emitted.decode())["source"]
        assert "ct.extract(" in src_cla
        call_cla.launch()
        torch.cuda.synchronize(device)
        # Verify final state exactness: sum_t k[b,t,d] * v[b,t,dv]
        fc_ref = torch.einsum("btd,btv->bdv", kc, vc)
        torch.testing.assert_close(fc, fc_ref, rtol=1e-3, atol=1e-3)

        qr = torch.randn((2, 64, 8, 64), generator=gen, device=device, dtype=torch.bfloat16)
        kr = torch.randn((2, 64, 8, 64), generator=gen, device=device, dtype=torch.bfloat16)
        vr = torch.randn((2, 64, 8, 64), generator=gen, device=device, dtype=torch.bfloat16)
        gam = torch.full((8,), 0.95, device=device, dtype=torch.float32)
        outr = torch.empty_like(vr)
        finr = torch.empty((2, 8, 64, 64), device=device, dtype=torch.float32)
        fn_ret = I.jit(group7_cpu.retention_recurrence, backend="cutile", library=library)
        call_ret = fn_ret.prepare(qr, kr, vr, gam, 0.125, outputs=(outr, finr))
        src_ret = json.loads(call_ret.artifact.emitted.decode())["source"]
        assert "ct.extract(" in src_ret and "ct.broadcast_to(ct.reshape(ct.extract(" in src_ret
        call_ret.launch()
        torch.cuda.synchronize(device)
        # Reference check on final state with constant decay 0.95
        decays = (0.95 ** torch.arange(63, -1, -1, device=device, dtype=torch.float32))[None, :, None, None]
        finr_ref = torch.einsum("bthk,bthv->bhkv", kr.float() * decays, vr.float())
        torch.testing.assert_close(finr, finr_ref, rtol=2e-2, atol=2e-2)

    # 3. Resident multi-reduction rematerialization & immediate disjoint stores (group_input_gradient)
    B_g, C_g, S_g, G_g = 4, 32, 256, 4
    xg = torch.randn((B_g, C_g, S_g), generator=gen, device=device, dtype=torch.float16)
    dyg = torch.randn((B_g, C_g, S_g), generator=gen, device=device, dtype=torch.float16)
    gaing = torch.randn((C_g,), generator=gen, device=device, dtype=torch.float16)
    meang = torch.randn((B_g, G_g), generator=gen, device=device, dtype=torch.float16)
    invg = torch.randn((B_g, G_g), generator=gen, device=device, dtype=torch.float16)
    dxg = torch.empty_like(xg)
    k_dx = examples.group_norm_backward.steps[3][1]
    fn_dx = I.jit(k_dx, backend=exit_name, library=library)
    call_dx = fn_dx.prepare(xg, dyg, gaing, meang, invg, 1.0 / (8 * S_g), outputs=(dxg,))
    call_dx.launch()
    torch.cuda.synchronize(device)
    xf = xg.float().reshape(B_g, G_g, 8, S_g)
    dyf = dyg.float().reshape(B_g, G_g, 8, S_g)
    gf = gaing.float().reshape(1, G_g, 8, 1)
    nf = (xf - meang.float()[:, :, None, None]) * invg.float()[:, :, None, None]
    uf = dyf * gf
    tot = uf.sum(dim=(2, 3), keepdim=True) / (8 * S_g)
    proj = (nf * uf).sum(dim=(2, 3), keepdim=True) / (8 * S_g)
    dx_ref = (invg.float()[:, :, None, None] * (uf - tot - nf * proj)).reshape(B_g, C_g, S_g).to(torch.float16)
    torch.testing.assert_close(dxg, dx_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("exit_name", ["triton", "cutile"])
def test_radix_pow2_l1scatter_qknr_index_add_and_awq_loops(exit_name):
    """Regression: pow2 floordivsi shift fast path + zero-atomic L1 scatter, narrow-row non-remat/occ4, index_add occ, AWQ stepped K-loop & SM90 row rect_read."""
    library = _require_library()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA device")
    if exit_name == "cutile":
        pytest.importorskip("cuda.tile")
    import numpy as np
    import intentdsl as I
    from intentdsl import group7, group7_cpu, group7_dsa

    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    sm = major * 10 + minor
    gen = torch.Generator(device=device).manual_seed(41)

    # 1. radix_pass_rows: dense duplicate bins + non-divisible N=100 for pow2 (1, 256) and non-pow2 (3) divisors
    keys_dup = torch.randint(0, 100000, (10, 100), generator=gen, device=device, dtype=torch.int32)
    keys_dup[:, :20] = 768
    keys_dup[:, 20:35] = 769
    vals = torch.arange(100, device=device, dtype=torch.int32).unsqueeze(0).expand(10, 100).contiguous()
    fn_rad = I.jit(group7_cpu.radix_pass_rows, backend=exit_name, library=library)
    for divisor in (1, 256, 3):
        hist = torch.zeros((10, 256), device=device, dtype=torch.int32)
        cur = torch.zeros((10, 256), device=device, dtype=torch.int32)
        call_rad = fn_rad.prepare(keys_dup, vals, hist, cur, divisor)
        art_rad = json.loads(call_rad.artifact.emitted.decode())
        src_rad = art_rad["source"]
        assert ">>" in src_rad and "//" in src_rad
        if exit_name == "cutile":
            # Zero-atomic L1 scatter: only histogram relaxed add + final acq_rel flush remain (no cursor atomics)
            assert src_rad.count("ct.atomic_add(") == 2
        else:
            warps = {r["native"]["num_warps"] for r in art_rad["rows"]}
            assert {2, 4, 8}.issubset(warps)
        call_rad.launch()
        torch.cuda.synchronize(device)
        _, _, offsets, out_keys, out_vals = call_rad.result()
        k_np = keys_dup.cpu().numpy()
        v_np = vals.cpu().numpy()
        digits = (k_np // divisor) % 256
        order = np.argsort(digits, axis=-1, kind="stable")
        assert torch.equal(out_keys.cpu(), torch.from_numpy(np.take_along_axis(k_np, order, axis=-1)))
        assert torch.equal(out_vals.cpu(), torch.from_numpy(np.take_along_axis(v_np, order, axis=-1)))
        for r in range(10):
            exp_off = np.zeros(257, dtype=np.int32)
            np.cumsum(np.bincount(digits[r], minlength=256), out=exp_off[1:])
            assert torch.equal(offsets[r].cpu(), torch.from_numpy(exp_off))

    # 2. qk_norm_rope: narrow width=64 must not double-load inputs via rematWrites; cuTile offers occupancy=4
    B, S = 2, 128
    q = torch.randn((B, 32, S, 128), generator=gen, device=device, dtype=torch.bfloat16)
    k = torch.randn((B, 8, S, 128), generator=gen, device=device, dtype=torch.bfloat16)
    wq = torch.ones((128,), device=device, dtype=torch.bfloat16)
    wk = torch.ones((128,), device=device, dtype=torch.bfloat16)
    cos = torch.ones((S, 64), device=device, dtype=torch.bfloat16)
    sin = torch.zeros((S, 64), device=device, dtype=torch.bfloat16)
    qo = torch.empty_like(q)
    ko = torch.empty_like(k)
    fn_qknr = I.jit(group7.qk_norm_rope, backend=exit_name, library=library)
    call_qknr = fn_qknr.prepare(q, k, wq, wk, cos, sin, 1.0 / 128.0, 1e-6, outputs=(qo, ko))
    art_qknr = json.loads(call_qknr.artifact.emitted.decode())
    src_qknr = art_qknr["source"]
    if exit_name == "triton":
        assert src_qknr.count("tl.load(") == 16
    else:
        assert src_qknr.count("ct.load(") == 16
        occs = {r["native"]["occupancy"] for r in art_qknr["rows"]}
        assert 4 in occs
    call_qknr.launch()
    torch.cuda.synchronize(device)
    qf = q.float()
    q_ref = (qf * torch.rsqrt(qf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    torch.testing.assert_close(qo, q_ref, rtol=2e-2, atol=2e-2)

    # 3. index_add_accumulate: cuTile emits explicit occupancy/opt_level hints for tile_atomic_add
    idx = torch.randint(0, 64, (256,), generator=gen, device=device, dtype=torch.int32)
    src_ia = torch.randn((256, 128), generator=gen, device=device, dtype=torch.float32)
    dst_ia = torch.zeros((64, 128), device=device, dtype=torch.float32)
    fn_ia = I.jit(group7_cpu.index_add_accumulate, backend=exit_name, library=library)
    call_ia = fn_ia.prepare(idx, src_ia, dst_ia)
    art_ia = json.loads(call_ia.artifact.emitted.decode())
    if exit_name == "cutile":
        assert any(r["native"].get("occupancy") == 4 and r["native"].get("opt_level") == 3 for r in art_ia["rows"])
    call_ia.launch()
    torch.cuda.synchronize(device)
    ref_ia = torch.zeros((64, 128), device=device, dtype=torch.float32).index_add_(0, idx.long(), src_ia)
    torch.testing.assert_close(dst_ia, ref_ia, rtol=1e-4, atol=1e-4)

    # 4. awq_int4_gemm: fused TMA path (M=32, K=1024, N=128 -> mBlocks=1, kPanels>=8):
    # Triton promotes stepped TMA K-loop; SM90 cuTile rewrites aligned 1-row qz/sc to ct.load
    M, N, K, G = 32, 128, 1024, 8
    x_awq = (torch.randn((M, K), generator=gen, device=device) * 0.25).to(torch.float16)
    shifts = [0, 16, 4, 20, 8, 24, 12, 28]
    w_codes = torch.randint(0, 16, (K, N), generator=gen, device=device, dtype=torch.int32)
    z_codes = torch.randint(0, 16, (G, N), generator=gen, device=device, dtype=torch.int32)
    qw = torch.zeros((K, N // 8), device=device, dtype=torch.int32)
    qz = torch.zeros((G, N // 8), device=device, dtype=torch.int32)
    for j, sh in enumerate(shifts):
        qw |= (w_codes[:, j::8] & 0xF) << sh
        qz |= (z_codes[:, j::8] & 0xF) << sh
    sc = (torch.rand((G, N), generator=gen, device=device) * 0.5 + 0.1).to(torch.float16)
    fn_awq = I.jit(group7_dsa.awq_int4_gemm, backend=exit_name, library=library)
    call_awq = fn_awq.prepare(x_awq, qw, sc, qz)
    src_awq = json.loads(call_awq.artifact.emitted.decode())["source"]
    if exit_name == "triton":
        assert "range(0,1024," in src_awq
    elif sm == 90:
        assert "ct.gather(" not in src_awq and "allow_tma=False" in src_awq
    call_awq.launch()
    torch.cuda.synchronize(device)
    w_ref = ((w_codes - z_codes.repeat_interleave(128, dim=0)).to(torch.float16) * sc.repeat_interleave(128, dim=0)).to(torch.float32)
    expected_awq = (x_awq.to(torch.float32) @ w_ref).to(torch.float16)
    torch.testing.assert_close(call_awq.result(), expected_awq, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("exit_name", ["triton", "cutile"])
def test_split_quant_contraction_cost_model_and_triton_tuner(exit_name):
    """Regression: cost-model split-vs-fused grouped-RHS contraction (vendor + library + L2/tail boundaries + CUDAGraph) and Triton median CUDA-event tuner."""
    library = _require_library()
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA device")
    if exit_name == "cutile":
        pytest.importorskip("cuda.tile")
    from statistics import median
    import intentdsl as I
    from intentdsl import examples, group7_dsa

    device = torch.device("cuda", torch.cuda.current_device())
    major, minor = torch.cuda.get_device_capability(device)
    sm = major * 10 + minor
    gen = torch.Generator(device=device).manual_seed(73)
    shifts = [0, 16, 4, 20, 8, 24, 12, 28]

    def make_awq(M, K, N):
        G = K // 128
        x = (torch.randn((M, K), generator=gen, device=device) * 0.125).to(torch.float16)
        w_codes = torch.randint(0, 16, (K, N), generator=gen, device=device, dtype=torch.int32)
        z_codes = torch.randint(0, 16, (G, N), generator=gen, device=device, dtype=torch.int32)
        qw = torch.zeros((K, N // 8), device=device, dtype=torch.int32)
        qz = torch.zeros((G, N // 8), device=device, dtype=torch.int32)
        for j, sh in enumerate(shifts):
            qw |= (w_codes[:, j::8] & 0xF) << sh
            qz |= (z_codes[:, j::8] & 0xF) << sh
        sc = (torch.rand((G, N), generator=gen, device=device) * 0.25 + 0.05).to(torch.float16)
        w_ref = ((w_codes - z_codes.repeat_interleave(128, dim=0)).to(torch.float16) * sc.repeat_interleave(128, dim=0)).to(torch.float32)
        ref = (x.to(torch.float32) @ w_ref).to(torch.float16)
        return x, qw, sc, qz, ref

    # 1. Production shape 256x4096x4096-g128: verify split-vs-fused cost selection under vendor & library + CUDAGraph safety
    x, qw, sc, qz, ref = make_awq(256, 4096, 4096)
    for reuse in ("vendor", "library"):
        fn = I.jit(group7_dsa.awq_int4_gemm, backend=exit_name, library=library, configuration={"reuse": reuse})
        call = fn.prepare(x, qw, sc, qz)
        art = json.loads(call.artifact.emitted.decode())
        if exit_name == "triton" or sm == 90:
            assert art.get("pre_stage") is not None, f"expected split on {exit_name} sm{sm} reuse={reuse}"
            assert bool(art.get("library_gemm")) == (reuse == "library")
        else:
            # SM120 cuTile dequant store overhead favors fused at M=256
            assert art.get("pre_stage") is None
        call.launch()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(call.result(), ref, rtol=2e-2, atol=2e-2)
        # Verify CUDA Graph capture + replay stability (pre-allocated workspace, zero capture-time allocs)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call.enqueue()
        g.replay()
        torch.cuda.synchronize(device)
        torch.testing.assert_close(call.result(), ref, rtol=2e-2, atol=2e-2)
        del g

    # 2. Boundary cases: L2-overflow (M=16, K=8192, N=8192) and non-divisible (M=17, K=128, N=200) must fall back to fused
    x_ov = torch.zeros((16, 8192), device=device, dtype=torch.float16)
    qw_ov = torch.zeros((8192, 1024), device=device, dtype=torch.int32)
    sc_ov = torch.ones((64, 8192), device=device, dtype=torch.float16)
    qz_ov = torch.zeros((64, 1024), device=device, dtype=torch.int32)
    fn_lib = I.jit(group7_dsa.awq_int4_gemm, backend=exit_name, library=library, configuration={"reuse": "library"})
    art_ov = json.loads(fn_lib.prepare(x_ov, qw_ov, sc_ov, qz_ov).artifact.emitted.decode())
    assert art_ov.get("pre_stage") is None

    x_nd, qw_nd, sc_nd, qz_nd, ref_nd = make_awq(17, 128, 200)
    call_nd = fn_lib.prepare(x_nd, qw_nd, sc_nd, qz_nd)
    assert json.loads(call_nd.artifact.emitted.decode()).get("pre_stage") is None
    call_nd.launch()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(call_nd.result(), ref_nd, rtol=2e-2, atol=2e-2)

    # 3. Triton runtime median CUDA-event tuner: multi-row kernels gather 9 samples/candidate and select min median
    if exit_name == "triton":
        xg = torch.randn((32, 256, 1024), generator=gen, device=device, dtype=torch.float16)
        dyg = torch.randn((32, 256, 1024), generator=gen, device=device, dtype=torch.float16)
        mg = torch.randn((32, 32), generator=gen, device=device, dtype=torch.float16)
        ig = torch.randn((32, 32), generator=gen, device=device, dtype=torch.float16)
        dwg = torch.empty((256,), device=device, dtype=torch.float16)
        dbg = torch.empty((256,), device=device, dtype=torch.float16)
        fn_aff = I.jit(examples.group_affine_gradient, backend="triton", library=library)
        call_aff = fn_aff.prepare(xg, dyg, mg, ig, outputs=(dwg, dbg))
        call_aff.launch()
        torch.cuda.synchronize(device)
        rep = call_aff.tuning_report()
        trials = rep["trials_us"]
        assert len(trials) >= 2
        for entry in trials:
            assert len(entry["samples"]) == 9
        expected_winner = min(trials, key=lambda e: median(e["samples"]))["row_id"]
        assert rep["winner_row_id"] == expected_winner
