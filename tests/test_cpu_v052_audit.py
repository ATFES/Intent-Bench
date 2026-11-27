"""Minimal reproducers for CompilerInfra 0.5.2 CPU migration audit (audit-mig)."""

import json
import os
import re

import pytest


def _require_compiler():
    library = os.environ.get("INTENTDSL_LIBRARY")
    if not library:
        pytest.skip("requires INTENTDSL_LIBRARY")
    from compilerinfra import Compiler
    return library, Compiler


def test_mojo_simd_fma_tail_proof_uses_actual_loop_offset():
    """Item 1a: K=4097 with depth=64 must NOT pass proveTailAccess with offset=0."""
    library, Compiler = _require_compiler()
    torch = pytest.importorskip("torch")
    import intentdsl as I

    @I.kernel
    def gemm_k_tail(lhs: I.In[I.f32, ("M", "K")],
                    rhs: I.In[I.f32, ("K", "N")],
                    out: I.Out[I.f32, ("M", "N")]):
        out[:, :] = I.contract(lhs[:, :], rhs[:, :], axes=((1,), (0,)), acc=I.f32)

    def realized_for_k(k_dim: int) -> tuple[str, str]:
        opts = {
            "vector_bits": 512,
            "workers": 1,
            "onednn_runtime": False,
            "calibration": "x86-1c",
            "arguments": [
                {"shape": [8, k_dim], "strides": [k_dim, 1]},
                {"shape": [k_dim, 32], "strides": [32, 1]},
                {"shape": [8, 32], "strides": [32, 1]},
            ],
        }
        with Compiler(library) as base, base.derive({"exits": {"mojo": {"options": opts}}}) as cc:
            root = base.root_ir(gemm_k_tail.source)
            ir = cc.compile(root, exit="mojo", through="mojo@realized", text=True).decode()
            pkg, _ = cc.compile_runtime(root, exit="mojo")
            return ir, json.loads(pkg)["source"]

    ir_4096, _ = realized_for_k(4096)
    ir_4097, src_4097 = realized_for_k(4097)

    # K=4096 is an exact multiple of depth=64 -> arith.minsi elided.
    assert "mojo_row.simd_fma" in ir_4096
    assert "arith.minsi" not in ir_4096

    # K=4097 has last block starting at p=4096 -> p+64=4160 > 4097; MUST clamp to 4097!
    assert "mojo_row.simd_fma" in ir_4097
    assert "arith.minsi" in ir_4097
    assert "min(" in src_4097 and "4097" in src_4097

    # End-to-end numerical check at K=4097 on Mojo SIMD FMA path:
    # transposed LHS (strides=(1, 8)) has non-unit K stride, so Mojo naturally selects simd-fma!
    if os.environ.get("INTENT_MOJO"):
        torch.manual_seed(0)
        a = torch.randn(4097, 8, dtype=torch.float32).T
        b = torch.randn(4097, 32, dtype=torch.float32)
        c = I.jit(gemm_k_tail, backend="mojo").run(a, b)
        assert torch.allclose(c, a @ b, atol=1e-3, rtol=1e-3)


def test_mojo_owner_axis_non_multiple_fast_span_and_weft_transpose_tail():
    """Item 1b/1c: Mojo ownerAxis fast-span tail proof & Weft transpose tail elision."""
    library, Compiler = _require_compiler()
    torch = pytest.importorskip("torch")
    import intentdsl as I
    from intentdsl import examples

    # S=17 >= 16 (SIMD width) but 17 % 16 != 0 and B=3 > 1: offset=0 falsely proved
    # ExactInBounds on ownerAxis=1 and straddled C owners!
    bn_opts = {
        "vector_bits": 512,
        "workers": 1,
        "calibration": "x86-1c",
        "arguments": [
            {"shape": [3, 5, 17], "strides": [85, 17, 1]},
            {"shape": [5], "strides": [1]},
            {"shape": [5], "strides": [1]},
            {"shape": [5], "strides": [1]},
            {"shape": [5], "strides": [1]},
            {"shape": [3, 5, 17], "strides": [85, 17, 1]},
            {"shape": [5], "strides": [1]},
            {"shape": [5], "strides": [1]},
            None,
            None,
        ],
    }
    with Compiler(library) as base, base.derive({"exits": {"mojo": {"options": bn_opts}}}) as cc:
        bn_ir = cc.compile(
            base.root_ir(examples.batch_norm_training.source),
            exit="mojo", through="mojo@realized", text=True
        ).decode()
    # Non-multiple S=17 and S=100 across B=3 must retain runtime in-span guard (mojo_row.if)
    # and per-lane fallback (mojo_row.acc_insert_lane) without straddling C owners!
    assert "mojo_row.if" in bn_ir and "mojo_row.acc_insert_lane" in bn_ir
    s100_opts = dict(bn_opts)
    s100_opts["arguments"] = [
        {"shape": [3, 5, 100], "strides": [500, 100, 1]},
        {"shape": [5], "strides": [1]},
        {"shape": [5], "strides": [1]},
        {"shape": [5], "strides": [1]},
        {"shape": [5], "strides": [1]},
        {"shape": [3, 5, 100], "strides": [500, 100, 1]},
        {"shape": [5], "strides": [1]},
        {"shape": [5], "strides": [1]},
        None,
        None,
    ]
    with Compiler(library) as base, base.derive({"exits": {"mojo": {"options": s100_opts}}}) as cc:
        s100_ir = cc.compile(
            base.root_ir(examples.batch_norm_training.source),
            exit="mojo", through="mojo@realized", text=True
        ).decode()
    assert "mojo_row.if" in s100_ir and "mojo_row.acc_insert_lane" in s100_ir

    if os.environ.get("INTENT_MOJO"):
        for s_ext in (17, 100):
            torch.manual_seed(1 + s_ext)
            x = torch.randn(3, 5, s_ext, dtype=torch.float16)
            w = torch.randn(5, dtype=torch.float32)
            b = torch.randn(5, dtype=torch.float32)
            mm = torch.zeros(5, dtype=torch.float32)
            mv = torch.ones(5, dtype=torch.float32)
            y, mean, inv, _, _ = I.jit(examples.batch_norm, backend="mojo").run(
                x, w, b, mm.clone(), mv.clone(), 1e-5, 0.1
            )
            ref_mean = x.float().mean(dim=(0, 2))
            assert torch.allclose(mean, ref_mean, atol=2e-3, rtol=2e-3)

    def weft_transpose_ir(m: int, n: int, in_s0: int | None = None) -> str:
        opts = {
            "vector_bits": 256,
            "vector_registers": 32,
            "workers": 1,
            "calibration": "x86-1c",
            "arguments": [
                {"shape": [m, n], "strides": [in_s0 or n, 1]},
                {"shape": [n, m], "strides": [m, 1]},
            ],
        }
        with Compiler(library) as base, base.derive({"exits": {"weft": {"options": opts}}}) as cc:
            return cc.compile(
                base.root_ir(examples.transpose.source),
                exit="weft", through="weft@realized", text=True
            ).decode()

    ir_exact_tile = weft_transpose_ir(2048, 2048, in_s0=2048)
    ir_unaligned = weft_transpose_ir(2048, 1003, in_s0=2048)
    ir_ext_100 = weft_transpose_ir(2048, 100, in_s0=2048)
    # Exact tiles elide both vector-tail and scalar-tail inner task_for loops; N=1003 and N=100 retain them.
    assert 'iv = "g"' in ir_exact_tile and 'iv = "g"' in ir_unaligned
    assert ir_exact_tile.count("weft_host.task_for") == 2
    assert ir_unaligned.count("weft_host.task_for") == 4
    assert ir_ext_100.count("weft_host.task_for") == 4
    import importlib.util
    weft_ready = False
    if (importlib.util.find_spec("weft_core") is not None and
            importlib.util.find_spec("triton.cpu") is not None):
        try:
            from triton.cpu import target as weft_target
            weft_target("native")
            weft_ready = True
        except Exception:
            weft_ready = False
    if weft_ready:
        t_in = torch.randn(2048, 2048, dtype=torch.float16)[:, :1003]
        t_out = I.jit(examples.transpose, backend="weft").run(t_in)
        assert torch.equal(t_out, t_in.T.contiguous())


def test_mojo_simd_fma_and_vnni_dot4_typed_contracts_enforce_types_ranks_and_layout():
    """Item 2: Mojo FMA and VNNI TypedContracts reject untyped/wrong-signedness/bad-layout IR."""
    library, Compiler = _require_compiler()
    from intentdsl import examples
    import intentdsl as I

    @I.kernel
    def gemm_f32(lhs: I.In[I.f32, ("M", "K")],
                 rhs: I.In[I.f32, ("K", "N")],
                 out: I.Out[I.f32, ("M", "N")]):
        out[:, :] = I.contract(lhs[:, :], rhs[:, :], axes=((1,), (0,)), acc=I.f32)

    vnni_opts = {
        "vector_bits": 512,
        "workers": 1,
        "grouped_i8": True,
        "onednn_runtime": False,
        "calibration": "x86-1c",
        "arguments": [
            {"shape": [8, 64], "strides": [64, 1]},
            {"shape": [64, 32], "strides": [32, 1]},
            {"shape": [32], "strides": [1]},
            {"shape": [8, 32], "strides": [32, 1]},
        ],
    }
    fma_opts = {
        "vector_bits": 512,
        "workers": 1,
        "onednn_runtime": False,
        "calibration": "x86-1c",
        "arguments": [
            {"shape": [8, 64], "strides": [64, 1]},
            {"shape": [64, 32], "strides": [32, 1]},
            {"shape": [8, 32], "strides": [32, 1]},
        ],
    }

    with Compiler(library) as base:
        with base.derive({"exits": {"mojo": {"options": vnni_opts}}}) as cc_vnni:
            vnni_ir = cc_vnni.compile(
                base.root_ir(examples.int8_gemm.source),
                exit="mojo", through="mojo@realized", text=True
            ).decode()
            assert "mojo_row.vnni_dot4" in vnni_ir
            assert "(vector<16xi32>, vector<64xui8>, vector<64xi8>) -> vector<16xi32>" in vnni_ir

            # Valid VNNI u8x64 * s8x64 -> s32x16 passes verification to callable.
            cc_vnni.compile(vnni_ir.encode(), exit="mojo",
                            after="mojo@realized", through="mojo@callable", text=True)

            # Dummy index operands (previously accepted under Any!) must fail.
            bad_index = (
                vnni_ir
                .replace("-> vector<64xui8>", "-> index")
                .replace("(vector<16xi32>, vector<64xui8>, vector<64xi8>)",
                         "(vector<16xi32>, index, vector<64xi8>)")
            )
            with pytest.raises(Exception):
                cc_vnni.compile(bad_index.encode(), exit="mojo",
                                after="mojo@realized", through="mojo@callable", text=True)

            # Wrong signedness on LHS (i8 instead of ui8) must fail.
            bad_sign = (
                vnni_ir
                .replace("-> vector<64xui8>", "-> vector<64xi8>")
                .replace("(vector<16xi32>, vector<64xui8>, vector<64xi8>)",
                         "(vector<16xi32>, vector<64xi8>, vector<64xi8>)")
            )
            with pytest.raises(Exception):
                cc_vnni.compile(bad_sign.encode(), exit="mojo",
                                after="mojo@realized", through="mojo@callable", text=True)

            # Non-unit LHS K stride must fail layout verification.
            bad_stride = vnni_ir.replace("lhs_strides = array<i64: 64, 1>",
                                         "lhs_strides = array<i64: 1, 64>")
            with pytest.raises(Exception):
                cc_vnni.compile(bad_stride.encode(), exit="mojo",
                                after="mojo@realized", through="mojo@callable", text=True)

            # Unpacked RHS layout must fail layout verification.
            bad_pack = vnni_ir.replace('rhs_packing = "signed-dot4-panels"',
                                       'rhs_packing = "row-major"')
            with pytest.raises(Exception):
                cc_vnni.compile(bad_pack.encode(), exit="mojo",
                                after="mojo@realized", through="mojo@callable", text=True)

        with base.derive({"exits": {"mojo": {"options": fma_opts}}}) as cc_fma:
            fma_ir = cc_fma.compile(
                base.root_ir(gemm_f32.source),
                exit="mojo", through="mojo@realized", text=True
            ).decode()
            assert "mojo_row.simd_fma" in fma_ir
            assert "(vector<16xf32>, vector<16xf32>, vector<16xf32>) -> vector<16xf32>" in fma_ir

            # Valid SIMD FMA passes; dummy index or mismatched is_int fails.
            cc_fma.compile(fma_ir.encode(), exit="mojo",
                           after="mojo@realized", through="mojo@callable", text=True)
            bad_fma_int = fma_ir.replace("is_int = false", "is_int = true")
            with pytest.raises(Exception):
                cc_fma.compile(bad_fma_int.encode(), exit="mojo",
                               after="mojo@realized", through="mojo@callable", text=True)
            bad_fma_rank = fma_ir.replace("lhs_strides = array<i64: 64, 1>",
                                          "lhs_strides = array<i64: 64>")
            with pytest.raises(Exception):
                cc_fma.compile(bad_fma_rank.encode(), exit="mojo",
                               after="mojo@realized", through="mojo@callable", text=True)


def test_weft_host_scheduling_is_explicit_in_realized_ir():
    """Item 3: Stride choices and OpenMP binning are materialized in weft@realized IR."""
    library, Compiler = _require_compiler()
    from intentdsl import examples

    base_opts = {
        "vector_bits": 512,
        "vector_registers": 32,
        "workers": 8,
        "calibration": "x86",
    }

    def weft_realized(src: str, views: list[tuple[list[int], list[int]]]) -> str:
        opts = dict(base_opts)
        opts["arguments"] = [{"shape": s, "strides": t} for s, t in views]
        with Compiler(library) as base, base.derive({"exits": {"weft": {"options": opts}}}) as cc:
            return cc.compile(
                base.root_ir(src), exit="weft", through="weft@realized", text=True
            ).decode()

    # Routed GEMM (expert_projection): compare unit-stride vs strided output.
    routed_kernel = examples.expert_projection
    ir_contig = weft_realized(routed_kernel.source, [
        ([16, 64], [64, 1]),
        ([5], [1]),
        ([32], [1]),
        ([4, 32, 64], [2048, 64, 1]),
        ([16, 2, 32], [64, 32, 1]),
    ])
    ir_strided = weft_realized(routed_kernel.source, [
        ([16, 64], [64, 1]),
        ([5], [1]),
        ([32], [1]),
        ([4, 32, 64], [2048, 64, 1]),
        ([16, 2, 32], [128, 64, 2]),
    ])
    for banned in (
        "host_scatter_routed",
        "host_parallel_tiles",
        "host_parallel_row_intervals",
        "host_parallel_transpose_tiles",
        "host_parallel_key_ranges",
        "host_parallel_owners",
    ):
        assert banned not in ir_contig
        assert banned not in ir_strided

    # Contiguous routed scatter chooses host_span_op memcpy in IR; strided chooses host_for + host_elem_op.
    assert "weft_host.host_span_op" in ir_contig
    assert 'dst_expr = "out"' in ir_contig and 'src_expr = "from"' in ir_contig
    assert 'dst_expr = "out+j*4LL"' in ir_strided and 'src_expr = "from+j*2LL"' in ir_strided

    # Row interval OpenMP binning is explicit in weft@realized IR.
    row_ir = weft_realized(examples.softmax.source, [
        ([64, 256], [256, 1]),
        ([64, 256], [256, 1]),
    ])
    assert "host_parallel_row_intervals" not in row_ir
    assert "weft_host.host_omp_for" in row_ir
    assert 'num_threads = "bins"' in row_ir
    assert "weft_host.host_while" in row_ir


def test_mojo_and_weft_zero_custom_print_rules():
    """Item 4: Both Mojo and Weft exits have zero custom print rules."""
    library, Compiler = _require_compiler()
    with Compiler(library) as base:
        desc = base.describe()
    exits = desc.get("exits", {})
    for ex_name in ("mojo", "weft"):
        assert ex_name in exits
        pr = exits[ex_name].get("print_rules", {})
        assert pr.get("custom_count", -1) == 0, f"{ex_name} still has custom print rules: {pr}"
        assert pr.get("custom", None) == []


def test_cpu_v053_address_op_interface_rejects_stride_product_overflow():
    """0.5.3 AddressOpInterface: 65537 x 65536 overflows signed i32 and must remain 64-bit."""
    library, Compiler = _require_compiler()
    import intentdsl as I

    @I.kernel
    def gemm_addr(lhs: I.In[I.f32, ("M", "K")],
                  rhs: I.In[I.f32, ("K", "N")],
                  out: I.Out[I.f32, ("M", "N")]):
        out[:, :] = I.contract(lhs[:, :], rhs[:, :], axes=((1,), (0,)), acc=I.f32)

    small_args = [
        {"shape": [64, 64], "strides": [64, 1]},
        {"shape": [64, 64], "strides": [64, 1]},
        {"shape": [64, 64], "strides": [64, 1]},
    ]
    overflow_args = [
        {"shape": [65537, 65537], "strides": [65537, 1]},
        {"shape": [65537, 65536], "strides": [65536, 1]},
        {"shape": [65537, 65536], "strides": [65536, 1]},
    ]
    with Compiler(library) as base:
        root = base.root_ir(gemm_addr.source)
        for ex_name, addr_op in (("mojo", "mojo_row.address"), ("weft", "weft_host.address")):
            small_opts = {"vector_bits": 512, "workers": 1, "onednn_runtime": False,
                          "calibration": "x86-1c", "arguments": small_args}
            over_opts = {"vector_bits": 512, "workers": 1, "onednn_runtime": False,
                         "calibration": "x86-1c", "arguments": overflow_args}
            with base.derive({"exits": {ex_name: {"options": small_opts}}}) as cc_ok:
                ok_ir = cc_ok.compile(root, exit=ex_name, through=f"{ex_name}@callable", text=True).decode()
                assert addr_op in ok_ir and "index_bits = 32 : i32" in ok_ir
            with base.derive({"exits": {ex_name: {"options": over_opts}}}) as cc_ov:
                ov_ir = cc_ov.compile(root, exit=ex_name, through=f"{ex_name}@callable", text=True).decode()
                found = False
                for line in ov_ir.splitlines():
                    if addr_op in line and "65536" in line and "index_bits = 64 : i32" in line:
                        found = True
                        forged = ov_ir.replace(line, line.replace("index_bits = 64 : i32", "index_bits = 32 : i32"), 1)
                        with pytest.raises(Exception):
                            cc_ov.compile(forged.encode(), exit=ex_name,
                                          after=f"{ex_name}@realized", through=f"{ex_name}@callable", text=True)
                        break
                assert found, f"{ex_name}: expected un-narrowed 64-bit address on 65537x65536 overflow"


def test_cpu_owner_space_extent_100_vs_128_tail_proof():
    """0.5.3 IndexedRegionOpInterface + TiledAccessOpInterface on cpu.owner_space (extent=100 vs 128)."""
    library, Compiler = _require_compiler()
    from intentdsl import examples

    def attn_opts(hq: int, out_hq: int | None = None) -> dict:
        hk = max(1, (hq + 3) // 4)
        ohq = hq if out_hq is None else out_hq
        return {
            "vector_bits": 512,
            "vector_registers": 32,
            "workers": 1,
            "calibration": "x86-1c",
            "arguments": [
                {"shape": [1, hq, 1, 64], "strides": [hq * 64, 64, 64, 1]},
                {"shape": [1, hk, 512, 64], "strides": [hk * 32768, 32768, 64, 1]},
                {"shape": [1, hk, 512, 64], "strides": [hk * 32768, 32768, 64, 1]},
                {"shape": [1, ohq, 1, 64], "strides": [ohq * 64, 64, 64, 1]},
                None,
            ],
        }

    with Compiler(library) as base:
        root = base.root_ir(examples.attention.source)
        # HQ=128 is divisible by packed cpu.owner_batch=4 -> passes proveTailAccess.
        with base.derive({"exits": {"mojo": {"options": attn_opts(128)}}}) as cc128:
            tasked_128 = cc128.compile(root, exit="mojo", through="cpu@tasked", text=True).decode()
            assert "cpu.owner_batch = 4 : i64" in tasked_128
            cc128.compile(tasked_128.encode(), exit="mojo", after="cpu@tasked", through="mojo@callable", text=True)

        # HQ=102 (not divisible by 4) rejects batch=4 and falls back to batch<=2.
        with base.derive({"exits": {"mojo": {"options": attn_opts(102)}}}) as cc102:
            tasked_102 = cc102.compile(root, exit="mojo", through="cpu@tasked", text=True).decode()
            assert "cpu.owner_batch = 4 : i64" not in tasked_102

        # Forging output view (%arg3) HQ extent to 100 while keeping packed cpu.owner_batch=4
        # MUST be rejected by cpu.owner_space IndexedRegionOpInterface + proveTailAccess(cpu.slice_write)!
        with base.derive({"exits": {"mojo": {"options": attn_opts(128, out_hq=100)}}}) as cc_forged:
            forged = tasked_128.replace(
                "%arg3: memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>> {cpu.shape = array<i64: 1, 128, 1, 64>, cpu.strides = array<i64: 8192, 64, 64, 1>}",
                "%arg3: memref<?x?x?x?xbf16, strided<[?, ?, ?, ?], offset: ?>> {cpu.shape = array<i64: 1, 100, 1, 64>, cpu.strides = array<i64: 6400, 64, 64, 1>}",
            )
            with pytest.raises(Exception, match="packed cpu.owner_batch tile exceeds proven owner-axis extent"):
                cc_forged.compile(forged.encode(), exit="mojo", after="cpu@tasked", through="mojo@callable", text=True)


def test_csr_spmv_reassoc_vs_strict_ordered_sum():
    """Reassociable segment_sum vector-accumulates while strict ordered_sum preserves left-to-right order."""
    library, Compiler = _require_compiler()
    from intentdsl import examples

    strict_source = examples.csr_spmv.source.replace(b"I.segment_sum(", b"I.ordered_sum(")
    assert strict_source != examples.csr_spmv.source

    csr_args = [
        {"shape": [32769], "strides": [1]},
        {"shape": [1048576], "strides": [1]},
        {"shape": [1048576], "strides": [1]},
        {"shape": [32768], "strides": [1]},
        {"shape": [32768], "strides": [1]},
    ]
    base_opts = {
        "vector_bits": 512,
        "vector_registers": 32,
        "workers": 8,
        "calibration": "x86",
        "arguments": csr_args,
    }
    with Compiler(library) as base:
        reassoc_root = base.root_ir(examples.csr_spmv.source)
        strict_root = base.root_ir(strict_source)

        with base.derive({"exits": {"mojo": {"options": base_opts}}}) as cc_m:
            m_reassoc = cc_m.compile(reassoc_root, exit="mojo", through="mojo@realized", text=True).decode()
            m_strict = cc_m.compile(strict_root, exit="mojo", through="mojo@realized", text=True).decode()
            assert "mojo_row.gather_slice" in m_reassoc and "mojo_row.lane_sum" in m_reassoc
            assert "mojo_row.lane_sum" not in m_strict

        with base.derive({"exits": {"weft": {"options": base_opts}}}) as cc_w:
            w_tasked = cc_w.compile(reassoc_root, exit="weft", through="cpu@tasked", text=True).decode()
            w_reassoc = cc_w.compile(reassoc_root, exit="weft", through="weft@realized", text=True).decode()
            w_strict = cc_w.compile(strict_root, exit="weft", through="weft@realized", text=True).decode()
            assert "cpu.segment_width = 16 : i64" in w_tasked
            assert "weft_host.task_reduce" in w_reassoc
            assert "!weft_kernel.value<f32,[16],[2]>" in w_reassoc
            assert "weft_host.task_reduce" not in w_strict
            # Checkpoint round-trip rejects invalid segment_width on cpu@tasked.
            forged = w_tasked.replace("cpu.segment_width = 16 : i64", "cpu.segment_width = 32 : i64")
            with pytest.raises(Exception):
                cc_w.compile(forged.encode(), exit="weft", after="cpu@tasked", through="weft@realized", text=True)

        # Short average segment length (nnz=8) selects segment_width=8 via CPU partition cost model.
        short_opts = dict(base_opts)
        short_opts["arguments"] = [
            {"shape": [32769], "strides": [1]},
            {"shape": [262144], "strides": [1]},
            {"shape": [262144], "strides": [1]},
            {"shape": [32768], "strides": [1]},
            {"shape": [32768], "strides": [1]},
        ]
        with base.derive({"exits": {"weft": {"options": short_opts}}}) as cc_ws:
            ws_tasked = cc_ws.compile(reassoc_root, exit="weft", through="cpu@tasked", text=True).decode()
            ws_realized = cc_ws.compile(reassoc_root, exit="weft", through="weft@realized", text=True).decode()
            assert "cpu.segment_width = 8 : i64" in ws_tasked
            assert "weft_host.task_reduce" in ws_realized
            assert "!weft_kernel.value<f32,[8],[2]>" in ws_realized


def test_weft_attention_uses_runtime_dv_loop():
    """Weft causal GQA attention uses a runtime dv0 task_for loop instead of 4x unrolled Pass 2b."""
    library, Compiler = _require_compiler()
    from intentdsl import examples

    opts = {
        "vector_bits": 512,
        "vector_registers": 32,
        "workers": 8,
        "calibration": "x86",
        "arguments": [
            {"shape": [2, 8, 256, 64], "strides": [131072, 16384, 64, 1]},
            {"shape": [2, 2, 256, 64], "strides": [32768, 16384, 64, 1]},
            {"shape": [2, 2, 256, 64], "strides": [32768, 16384, 64, 1]},
            {"shape": [2, 8, 256, 64], "strides": [131072, 16384, 64, 1]},
            None,
        ],
    }
    with Compiler(library) as base, base.derive({"exits": {"weft": {"options": opts}}}) as cc:
        ir = cc.compile(base.root_ir(examples.attention.source), exit="weft", through="weft@realized", text=True).decode()
    assert "weft_host.task_for" in ir
    # Single runtime dv0 loop produces 1 PV outer_contract per pass rather than 4 unrolled copies.
    assert ir.count("weft_host.task_outer_contract") <= 4

    # Alternating generated kernels with different AMX tile configurations
    # must clear each worker's cached configuration before returning.
    family = "levels/execution/cpu::ordered-owners"
    brgemm = dict(opts, arch="x86_64", weft_version="0.2.0", amx_bf16=True,
                  private_stack_bytes=65536, l2_cache_bytes=2097152,
                  weft_brgemm_available=True, weft_brgemm_bf16=True,
                  weft_brgemm_runtime="SEQ", denormal_fp_math="preserve-sign")
    brgemm["arguments"] = [dict(arg, alias_group=i) if arg else None
                           for i, arg in enumerate(opts["arguments"])]
    with Compiler(library) as base:
        root = base.root_ir(examples.attention.source)
        decision = base.describe()["configuration"]["nodes"]["cpu"]["decisions"][family]
        forced = dict(decision, candidate="shared-256x64-scores-brgemm-64-pv")
        with base.derive({"exits": {"weft": {"options": brgemm}},
                          "nodes": {"cpu": {"decisions": {family: forced}}}}) as cc:
            managed = cc.compile(root, exit="weft", through="weft@realized", text=True).decode()
        assert 'engine = "onednn-brgemm-bf16"' in managed
        team = managed.index('"weft_host.host_omp_parallel"')
        assert "release_hw_ctx = true" in managed[team:].splitlines()[0]
        assert managed.index('"weft_host.host_collect_task_status"') > team

        rvv = dict(opts, arch="riscv64", core="sg2044", calibration="sg2044", vector_bits=128)
        with base.derive({"exits": {"weft": {"options": rvv}}}) as cc:
            vector = json.loads(cc.compile(root, exit="weft"))
        assert "weft_kernel_release_hw_context" not in vector["host"]
        assert "weft_kernel_take_status" not in vector["host"]


def test_gemm_nt_tls_panel_streaming_and_weft_assume_bounds():
    """gemm_nt 2048x2048x2048 f16 selects 12x64 TLS panel streaming and emits positive+upper tile assumes on Weft."""
    library, Compiler = _require_compiler()
    from intentdsl import group7_dsa

    args = [
        {"shape": [2048, 2048], "strides": [2048, 1]},
        {"shape": [2048, 2048], "strides": [2048, 1]},
        {"shape": [2048, 2048], "strides": [2048, 1]},
    ]
    opts = {
        "vector_bits": 512,
        "vector_registers": 32,
        "workers": 8,
        "onednn_runtime": True,
        "weft_version": "0.2.0",
        "weft_brgemm_available": True,
        "weft_brgemm_runtime": "SEQ",
        "weft_brgemm_f32": True,
        "calibration": "x86",
        "arguments": args,
    }
    with Compiler(library) as base:
        root = base.root_ir(group7_dsa.gemm_nt.source)
        with base.derive({"exits": {"mojo": {"options": opts}}}) as cc_m:
            m_ir = cc_m.compile(root, exit="mojo", through="mojo@realized", text=True).decode()
            assert 'placement = "tls"' in m_ir
            assert 'name = "b_grp"' in m_ir and 'name = "a_tile"' in m_ir
            assert 'name = "left"' not in m_ir and 'name = "rhs"' not in m_ir
            assert "tile = array<i64: 12, 64," in m_ir

        with base.derive({"exits": {"weft": {"options": opts}}}) as cc_w:
            w_ir = cc_w.compile(root, exit="weft", through="weft@realized", text=True).decode()
            assert 'placement = "tls"' in w_ir
            assert 'name = "b_grp"' in w_ir and 'name = "a_tile"' in w_ir
            assert 'name = "left"' not in w_ir and 'name = "rhs"' not in w_ir
            assert w_ir.count("weft_host.task_assume") >= 4
            assert 'predicate = "gt"' in w_ir and 'predicate = "le"' in w_ir

        # Quantized decoded-f32-panels (awq_int4_gemm, K=4096) allocates M*K left buffer
        # and must emit row_stride=4096 on widen-lhs-f32 (never padded 4112).
        awq_opts = dict(opts)
        awq_opts["arguments"] = [
            {"shape": [256, 4096], "strides": [4096, 1]},
            {"shape": [4096, 512], "strides": [512, 1]},
            {"shape": [32, 4096], "strides": [4096, 1]},
            {"shape": [32, 512], "strides": [512, 1]},
            {"shape": [256, 4096], "strides": [4096, 1]},
        ]
        awq_root = base.root_ir(group7_dsa.awq_int4_gemm.source)
        with base.derive({"exits": {"mojo": {"options": awq_opts}}}) as cc_awq:
            awq_ir = cc_awq.compile(awq_root, exit="mojo", through="mojo@realized", text=True).decode()
            assert "row*4096LL" in awq_ir and "row*4112LL" not in awq_ir


def test_global_numerical_options_merge_in_kir_canonical():
    """CompilerInfra configuration.numerical merges into kir@canonical like fpmath_mode=fast."""
    library, Compiler = _require_compiler()
    from intentdsl import examples

    base_opts = {
        "vector_bits": 512,
        "vector_registers": 32,
        "workers": 8,
        "calibration": "x86-1c",
        "onednn_runtime": True,
        "arguments": [
            {"shape": [4, 256, 256], "strides": [65536, 256, 1]},
            {"shape": [4, 256, 256], "strides": [65536, 256, 1]},
            {"shape": [4, 256, 256], "strides": [65536, 256, 1]},
        ],
    }
    with Compiler(library) as base:
        root_bg = base.root_ir(examples.batched_gemm.source)
        with base.derive({"exits": {"mojo": {"options": base_opts}}}) as cc_strict:
            ir_strict = cc_strict.compile(root_bg, exit="mojo", through="kir@canonical", text=True).decode()
            assert '"denormal-fp-math" = "ieee"' in ir_strict
        with base.derive({
            "numerical": {"fastmath": "fast", "denormal_mode": "preserve-sign"},
            "exits": {"mojo": {"options": base_opts}},
        }) as cc_num:
            ir_num = cc_num.compile(root_bg, exit="mojo", through="kir@canonical", text=True).decode()
            assert '"denormal-fp-math" = "preserve-sign"' in ir_num
        with base.derive({"exits": {"mojo": {"options": {**base_opts, "fpmath_mode": "fast"}}}}) as cc_fp:
            ir_fp = cc_fp.compile(root_bg, exit="mojo", through="kir@canonical", text=True).decode()
            assert ir_num == ir_fp


def test_weft_ordered_vectorizes_contiguous_inner_loops():
    """Weft Ordered vectorizes contiguous stride-1 reassociable reductions, independent writes, and pure scf.if."""
    library, Compiler = _require_compiler()
    from intentdsl import group7, group7_cpu

    base_opts = {
        "vector_bits": 512,
        "vector_registers": 32,
        "workers": 8,
        "calibration": "x86",
    }
    with Compiler(library) as base:
        kl_opts = {
            **base_opts,
            "arguments": [
                {"shape": [64, 128], "strides": [128, 1]},
                {"shape": [64, 128], "strides": [128, 1]},
                {"shape": [64, 128], "strides": [128, 1]},
                {"shape": [64], "strides": [1]},
            ],
        }
        with base.derive({"exits": {"weft": {"options": kl_opts}}}) as cc_kl:
            kl_ir = cc_kl.compile(
                base.root_ir(group7_cpu.log_softmax_kl.source),
                exit="weft", through="weft@realized", text=True,
            ).decode()
        assert kl_ir.count("weft_host.task_subview") >= 4
        assert kl_ir.count("weft_host.task_reduce") >= 3
        assert kl_ir.count("weft_host.task_commit") >= 2
        assert "weft_host.task_if" not in kl_ir
        assert "weft_host.task_select" in kl_ir

        ce_opts = {
            **base_opts,
            "arguments": [
                {"shape": [64, 256], "strides": [256, 1]},
                {"shape": [64], "strides": [1]},
                {"shape": [64], "strides": [1]},
                {"shape": [64], "strides": [1]},
                {"shape": [64, 256], "strides": [256, 1]},
            ],
        }
        with base.derive({"exits": {"weft": {"options": ce_opts}}}) as cc_ce:
            ce_ir = cc_ce.compile(
                base.root_ir(group7.cross_entropy.source),
                exit="weft", through="weft@realized", text=True,
            ).decode()
        assert ce_ir.count("weft_host.task_subview") >= 4
        assert ce_ir.count("weft_host.task_reduce") >= 2
        assert ce_ir.count("weft_host.task_commit") >= 2
        assert "weft_host.task_if" not in ce_ir


def test_weft_ordered_tiled_contraction_extensions():
    """Weft Ordered 2D outer_contract handles scf.if causal, post-sub scale, affine bias and output FTZ, and stack-budgeted subDV strips."""
    library, Compiler = _require_compiler()
    from intentdsl import group7

    def check_retained_scores(ir, tokens, batch):
        # A complete private window keeps one score per key and query. The
        # normalized value contraction reuses it without a second QK fold.
        lines = ir.splitlines()
        contracts = [line for line in lines if '"weft_host.task_contract"' in line]
        assert len(contracts) == 2
        assert "over = array<i64: 4>" in contracts[0]  # head dimension
        assert "over = array<i64: 6>" in contracts[1]  # keys
        buffers = [line.split("=")[0].strip() for line in lines
                   if '"weft_host.task_local_alloc"' in line
                   and f"elements = {tokens * batch} : i64" in line]
        assert len(buffers) == 1
        ptr = buffers[0]
        replay = [i for i, line in enumerate(lines) if f'"weft_host.task_ptr_load"({ptr},' in line]
        writes = [i for i, line in enumerate(lines) if f'"weft_host.task_ptr_store"({ptr},' in line]
        weight = re.search(r'task_contract"\((%\w+),', contracts[1])[1]
        weight_load = next(i for i in replay if lines[i].lstrip().startswith(f"{weight} = "))
        assert lines.index(contracts[0]) < writes[0] < replay[0]
        assert any(replay[0] < write < weight_load for write in writes)
        assert weight_load < lines.index(contracts[1])

    base_opts = {
        "vector_bits": 512,
        "vector_registers": 32,
        "workers": 8,
        "calibration": "x86",
    }
    attn_args = [
        {"shape": [2, 32, 2048, 128], "strides": [8388608, 262144, 128, 1]},
        {"shape": [2, 8, 2048, 128], "strides": [2097152, 262144, 128, 1]},
        {"shape": [2, 8, 2048, 128], "strides": [2097152, 262144, 128, 1]},
        {"shape": [2, 32, 2048, 128], "strides": [8388608, 262144, 128, 1]},
        None,
    ]
    alibi_args = [
        attn_args[0],
        attn_args[1],
        attn_args[2],
        {"shape": [32], "strides": [1]},
        attn_args[3],
        None,
    ]
    swa_args = [
        {"shape": [2, 32, 4096, 128], "strides": [16777216, 524288, 128, 1]},
        {"shape": [2, 8, 4096, 128], "strides": [4194304, 524288, 128, 1]},
        {"shape": [2, 8, 4096, 128], "strides": [4194304, 524288, 128, 1]},
        {"shape": [2, 32, 4096, 128], "strides": [16777216, 524288, 128, 1]},
        None,
        None,
    ]
    def compile_tiles(base, program, arguments):
        options = {**base_opts, "arguments": arguments}
        with base.derive({"exits": {"weft": {"options": options}}}) as cc:
            root = base.root_ir(program.source)
            tasked = cc.compile(root, exit="weft", through="cpu@tasked", text=True).decode()
            realized = cc.compile(root, exit="weft", through="weft@realized", text=True).decode()
        batch = int(re.search(r"cpu\.owner_batch = (\d+)", tasked)[1])
        panel = re.search(r"cpu\.vector_panel = \{([^}]+)\}", tasked)[1]
        geometry = {key: int(value) for key, value in re.findall(r"(\w+) = (\d+)", panel)}
        return realized, batch, geometry

    with Compiler(library) as base:
        unscaled_ir, batch, panel = compile_tiles(base, group7.attention_unscaled_max, attn_args)
        assert panel["score_slots"] == attn_args[1]["shape"][2]
        check_retained_scores(unscaled_ir, tokens=panel["score_slots"], batch=batch)
        assert 'engine = "auto"' in unscaled_ir
        assert f"!weft_kernel.value<bf16,[{batch},128],[3,4]>" in unscaled_ir
        assert f"elements = {batch * 128} : i64" in unscaled_ir  # tiled output (batch * DV)

        alibi_ir, batch, panel = compile_tiles(base, group7.alibi_attention, alibi_args)
        assert panel["score_slots"] == alibi_args[1]["shape"][2]
        check_retained_scores(alibi_ir, tokens=panel["score_slots"], batch=batch)
        # ALiBi's local exp2 enables flushing subnormal *outputs*.
        assert 'lit = "-126.0"' in alibi_ir
        assert 'lit = "-87.33654022216796875"' not in unscaled_ir
        assert 'kind = "exp2"' in alibi_ir

        swa_ir, batch, panel = compile_tiles(base, group7.sliding_window_attention, swa_args)
        assert swa_ir.count("weft_host.task_contract") >= 4
        assert f"!weft_kernel.value<f16,[{batch},128],[2,4]>" in swa_ir
        assert 0 < panel["value_width"] < 128
        assert f"!weft_kernel.value<f32,[{batch},{panel['value_width']}],[2,7]>" in swa_ir
        assert panel["token_tile"] <= panel["score_slots"] < swa_args[1]["shape"][2]
        assert f"elements = {batch * panel['score_slots']} : i64" in swa_ir


def test_weft_exp_ftz_preserves_small_normal_weights():
    library, _ = _require_compiler()
    pytest.importorskip("weft_core")
    torch = pytest.importorskip("torch")
    import intentdsl as I
    from intentdsl import examples

    q = torch.zeros((1, 4, 512, 128), dtype=torch.bfloat16)
    q[..., 0] = 1
    k = torch.zeros((1, 1, 512, 128), dtype=q.dtype)
    k[:, :, 1:, 0] = -1
    v = torch.zeros_like(k)
    v[:, :, 1:, :] = 2.0**120
    # exp(-87.332) is normal in binary32. The large value makes a wrongly
    # flushed weight visible in the final BF16 result.
    scale = 87.332
    kernel = I.jit(examples.attention_unscaled_max, library=library, backend="weft",
                   denormal_fp_math="preserve-sign", numerics="relaxed_normalization")
    from types import SimpleNamespace
    source = examples.attention_unscaled_max.source.replace(
        b"@I.kernel(", b'@I.kernel(denormal_fp_math="preserve-sign", ', 1)
    author = I.jit(SimpleNamespace(source=source), library=library, backend="weft")

    def panel_modes(program):
        with program.compilation_target(q, k, v, scale) as cc:
            task = cc.compile(cc.root_ir(program.source), exit="weft", through="cpu@tasked", text=True)
        return tuple(mark in task for mark in (b"cpu.panel_engine", b"cpu.pv_engine", b"cpu.worker_output_ftz"))

    assert panel_modes(author) == panel_modes(kernel)
    actual = author.run(q, k, v, scale)
    weight = torch.exp(torch.tensor(-scale)).bfloat16().float()
    expected = (torch.arange(512).float() * weight * 2.0**120).bfloat16()
    torch.testing.assert_close(actual[0, 0, :, 0], expected, atol=0, rtol=0)


@pytest.mark.parametrize("stage", ["input", "product"])
def test_weft_native_and_explicit_ieee_bf16_dot(stage):
    library, _ = _require_compiler()
    torch = pytest.importorskip("torch")
    pytest.importorskip("weft_core")
    import intentdsl as I
    from intentdsl import examples

    q = torch.zeros((1, 4, 256, 128), dtype=torch.bfloat16)
    k = torch.zeros((1, 1, 256, 128), dtype=q.dtype)
    v = torch.zeros_like(k)
    if stage == "input":
        q.view(torch.int16)[..., 0] = 0x0040  # BF16 2^-127
        k[:, :, ::2, 0] = 2.0**127
        v[:, :, ::2, :] = 1
        row, ieee, native = -1, 0x3F3B, 0x3F00
    else:
        q[..., 0] = 1
        k[:, :, 1, 0] = -0.6953125
        v[:, :, 1, :] = 2.0**-126
        row, ieee, native = 1, 0x002B, 0
    args = q, k, v, 1.0
    for options, bits in [
        ({}, native),
        ({"denormal_fp_math": "ieee"}, ieee),
        ({"float32_matmul_precision": "highest"}, ieee),
    ]:
        ker = I.jit(
            examples.attention_unscaled_max, backend="weft", library=library, **options
        )
        values, _, device = ker.binding.bind(args, None)
        caps = ker.target.options(ker.target.profile(device, values))
        if not (caps.get("amx_bf16") and caps.get("weft_brgemm_bf16")):
            pytest.skip("requires AMX BF16 and oneDNN BRGEMM")
        assert int(ker.run(*args)[0, 0, row, 0].view(torch.int16)) == bits
    reference = torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1),
        is_causal=True, scale=1.0,
    )
    assert int(reference[0, 0, row, 0].view(torch.int16)) == native
