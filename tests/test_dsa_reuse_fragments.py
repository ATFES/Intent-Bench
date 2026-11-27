"""Minimal reproducers for DSA ReuseFragments conservative replay & invalidation."""

import os
import pytest


ROW_IR_TEMPLATE = """module {{
  func.func @repro(%arg0: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg1: memref<?xf32, strided<[?], offset: ?>>, %arg2: memref<?x?xf32, strided<[?, ?], offset: ?>>, %arg3: f32) attributes {{block.configs = [{{id = 2064 : i32, payload = "16", rows = 16 : i32, width = 64 : i32}}], block.grid = ["ceildiv", "M", "rows"], block.guard = array<i64: 33, 64>, block.topology = "resident", "denormal-fp-math" = "ieee", kir.abi = "[{{\\22name\\22:\\22x\\22,\\22dtype\\22:\\22f32\\22,\\22dims\\22:[\\22M\\22,\\22N\\22],\\22direction\\22:\\22InOut\\22}},{{\\22name\\22:\\22gain\\22,\\22dtype\\22:\\22f32\\22,\\22dims\\22:[\\22N\\22],\\22direction\\22:\\22In\\22}},{{\\22name\\22:\\22out\\22,\\22dtype\\22:\\22f32\\22,\\22dims\\22:[\\22M\\22,\\22N\\22],\\22direction\\22:\\22Out\\22}},{{\\22name\\22:\\22s\\22,\\22dtype\\22:\\22f32\\22,\\22dims\\22:[],\\22direction\\22:\\22Scalar\\22}}]", kir.domain_arg = 0 : i64}} {{
    %0 = "block.pid"() {{axis = 0 : i32}} : () -> index
    %1 = "block.rows"(%0) {{step = "rows"}} : (index) -> tensor<?xindex>
    %2 = "block.zero"() : () -> index
    %3 = "block.columns"(%2) : (index) -> tensor<?xindex>
    %4 = "block.active"(%arg0, %1, %3) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, tensor<?xindex>, tensor<?xindex>) -> tensor<?x?xi1>
{body}
    return
  }}
}}
"""


def _compile_ir(ir_text: str, shapes_strides: list[tuple[list[int], list[int]]]) -> str:
    library = os.environ.get("INTENTDSL_LIBRARY")
    if not library:
        pytest.skip("requires an installed IntentDSL compiler")
    from compilerinfra import Compiler

    opts = dict(
        calibration="mlu370",
        clusters=4,
        cores=4,
        isa=372,
        nram_bytes=786432,
        wram_bytes=1048576,
        arguments=[dict(shape=s, strides=t) for s, t in shapes_strides],
    )
    with Compiler(library) as base, base.derive({"exits": {"bangc": {"options": opts}}}) as comp:
        out = comp.compile(
            ir_text.encode(),
            exit="bangc",
            after="block@realized",
            through="dsa@formed",
            text=True,
        )
    return out.decode()


def test_reuse_fragments_invalidates_across_intervening_store():
    """Before audit-mig fix: ReuseFragments bypassed proveValueReplay for block.load
    and did `continue` on Write-bearing MemoryEffectOpInterface ops without evicting
    cached reads, allowing a post-write read to reuse a pre-write stale fragment.
    After fix: both reads are preserved (count == 2).
    """
    row_body = """    %first = "block.load"(%arg0, %1, %3, %4) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, tensor<?xindex>, tensor<?xindex>, tensor<?x?xi1>) -> tensor<?x?xf32>
    %g = "block.load"(%arg1, %1, %3, %4) : (memref<?xf32, strided<[?], offset: ?>>, tensor<?xindex>, tensor<?xindex>, tensor<?x?xi1>) -> tensor<?x?xf32>
    %inc = "block.add"(%first, %g) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    "block.store"(%arg0, %1, %3, %4, %inc) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, tensor<?xindex>, tensor<?xindex>, tensor<?x?xi1>, tensor<?x?xf32>) -> ()
    %second = "block.load"(%arg0, %1, %3, %4) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, tensor<?xindex>, tensor<?xindex>, tensor<?x?xi1>) -> tensor<?x?xf32>
    %sum = "block.add"(%first, %second) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    "block.store"(%arg2, %1, %3, %4, %sum) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, tensor<?xindex>, tensor<?xindex>, tensor<?x?xi1>, tensor<?x?xf32>) -> ()"""
    ir_row = _compile_ir(
        ROW_IR_TEMPLATE.format(body=row_body),
        [([4, 64], [64, 1]), ([64], [1]), ([4, 64], [64, 1])],
    )
    assert ir_row.count('"block.load"(%arg0') == 2

    may_alias_body = """    %first = "block.load"(%arg0, %1, %3, %4) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, tensor<?xindex>, tensor<?xindex>, tensor<?x?xi1>) -> tensor<?x?xf32>
    "block.store"(%arg2, %1, %3, %4, %first) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, tensor<?xindex>, tensor<?xindex>, tensor<?x?xi1>, tensor<?x?xf32>) -> ()
    %second = "block.load"(%arg0, %1, %3, %4) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, tensor<?xindex>, tensor<?xindex>, tensor<?x?xi1>) -> tensor<?x?xf32>
    %sum = "block.add"(%first, %second) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    "block.store"(%arg0, %1, %3, %4, %sum) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, tensor<?xindex>, tensor<?xindex>, tensor<?x?xi1>, tensor<?x?xf32>) -> ()"""
    ir_may_alias = _compile_ir(
        ROW_IR_TEMPLATE.format(body=may_alias_body),
        [([4, 64], [64, 1]), ([64], [1]), ([4, 64], [64, 1])],
    )
    assert ir_may_alias.count('"block.load"(%arg0') == 2


def test_reuse_fragments_replays_proven_read_without_writes():
    """When proveValueReplay succeeds (no intervening writes or unknown ops),
    redundant block.load on the same SSA inputs is reused (count == 1).
    """
    safe_body = """    %first = "block.load"(%arg0, %1, %3, %4) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, tensor<?xindex>, tensor<?xindex>, tensor<?x?xi1>) -> tensor<?x?xf32>
    %sq = "block.mul"(%first, %first) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    %second = "block.load"(%arg0, %1, %3, %4) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, tensor<?xindex>, tensor<?xindex>, tensor<?x?xi1>) -> tensor<?x?xf32>
    %sum = "block.add"(%sq, %second) : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
    "block.store"(%arg2, %1, %3, %4, %sum) : (memref<?x?xf32, strided<[?, ?], offset: ?>>, tensor<?xindex>, tensor<?xindex>, tensor<?x?xi1>, tensor<?x?xf32>) -> ()"""
    ir_safe = _compile_ir(
        ROW_IR_TEMPLATE.format(body=safe_body),
        [([4, 64], [64, 1]), ([64], [1]), ([4, 64], [64, 1])],
    )
    assert ir_safe.count('"block.load"(%arg0') == 1


def test_mlu370_supply_cost_prefers_single_step_batch_over_multi_step_double_buffer():
    """Minimal reproducer for MLU370 supply cost misranking on small resident norms:
    Before calibration/model fix, rank-1 scalar ops were counted as N-lane vector
    ops, persistent prologue loads and tail-synced writeback were added to the
    double-buffer overlap pool, and per-step DMA round-trip latency was omitted,
    causing layer_norm (64x1024) to pick double-buffer-shared-16 over the faster
    single-step direct-shared-16-b4 while still preferring double-buffer-shared-16-b4
    at large M (4096x1024).
    """
    library = os.environ.get("INTENTDSL_LIBRARY")
    if not library:
        pytest.skip("requires an installed IntentDSL compiler")
    from compilerinfra import Compiler
    from intentdsl import examples as E

    def _selected_supply(rows: int) -> tuple[str, list[str], dict]:
        opts = dict(
            calibration="mlu370",
            clusters=4,
            cores=4,
            isa=372,
            nram_bytes=786432,
            sram_bytes=4194304,
            wram_bytes=1048576,
            fast_math=False,
            fpmath_mode="strict",
            arguments=[
                dict(alias_group=0, shape=[rows, 1024], strides=[1024, 1]),
                dict(alias_group=1, shape=[1024], strides=[1]),
                dict(alias_group=2, shape=[1024], strides=[1]),
                dict(alias_group=3, shape=[rows, 1024], strides=[1024, 1]),
                None,
                None,
            ],
        )
        with Compiler(library) as base, base.derive({"exits": {"bangc": {"options": opts}}}) as comp:
            _, report = comp.compile_measured(base.root_ir(E.layer_norm.source), exit="bangc")
        chosen = [
            ev["details"]
            for ev in report["events"]
            if ev.get("action") == "candidate"
            and ev.get("family") == "levels/storage/dsa::supply"
            and ev["details"].get("selected")
            and not ev["details"].get("eliminated")
        ]
        assert len(chosen) == 1
        row0 = next(iter(chosen[0]["rows"].values()))
        return chosen[0]["candidate"], chosen[0].get("missing", []), row0["terms"]

    small_cand, small_missing, small_terms = _selected_supply(64)
    assert small_missing == []
    assert small_cand == "direct-shared-16-b4"
    assert small_terms["shared_prologue"]["value"] > 0.0

    large_cand, large_missing, _ = _selected_supply(4096)
    assert large_missing == []
    assert large_cand == "double-buffer-shared-16-b4"


def test_strict_quotient_narrowing_keeps_negative_zero_and_subnormals():
    from types import SimpleNamespace
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_mlu")
    intentdsl = pytest.importorskip("intentdsl")
    if not torch.mlu.is_available():
        pytest.skip("requires a physical MLU")

    source = b"""@I.kernel
def divide(x: I.In[I.f32, ("M","N")], y: I.In[I.f32, ("M","N")], out: I.Out[I.f16, ("M","N")]):
    for i in I.parallel(x.dim(0)):
        out[i,:] = I.cast(x[i,:] / y[i,:], I.f16)
"""
    x = torch.tensor([-0.0, -(2.0**-26), -(2.0**-24), 2.0**-24] * 17).reshape(1, -1)
    divisor = torch.full_like(x, 1.25)
    kernel = intentdsl.jit(SimpleNamespace(source=source), backend="bangc",
                           fpmath_mode="strict", denormal_fp_math="ieee")
    result = kernel.run(x.to("mlu"), divisor.to("mlu")).cpu()
    assert torch.equal(result.view(torch.int16), (x / divisor).half().view(torch.int16))
