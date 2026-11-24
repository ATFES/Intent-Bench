"""Minimal reproducers for CompilerInfra 0.5.2 CPU migration audit (audit-mig)."""

import json
import os

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

