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
