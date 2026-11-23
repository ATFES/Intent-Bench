"""Compile valid strided views whose last offsets exceed signed i32."""

import json
import os

import pytest


@pytest.mark.parametrize("exit_name", ["bangc", "triton"])
def test_wide_record_and_scale_offsets(exit_name):
    library = os.environ.get("INTENTDSL_LIBRARY")
    if not library:
        pytest.skip("requires an installed IntentDSL compiler")
    from compilerinfra import Compiler
    from intentdsl import examples

    stride = 1 << 30
    if exit_name == "bangc":
        kernel = examples.q4_projection
        views = [([1, 1, 144], [1, 1, stride]), ([1, 256], [256, 1]), ([1], [1])]
        options = dict(calibration="mlu370", clusters=4, cores=4, isa=372,
                       nram_bytes=786432, wram_bytes=1048576)
    else:
        kernel = examples.fp8_gemm
        views = [([1, 3, 32], [96, 32, 1]), ([1, 3], [1, stride]),
                 ([3, 32, 1], [32, 1, 1]), ([3, 1], [stride, 1]), ([1, 1], [1, 1])]
        options = dict(sm=90, calibration="sm90", sm_count=132, warp_size=32,
                       max_threads=1024, registers=65536, shared_per_block=49152,
                       dynamic_shared_per_block=232448)
    options["arguments"] = [dict(shape=s, strides=t) for s, t in views]
    with Compiler(library) as base, base.derive({"exits": {exit_name: {"options": options}}}) as compiler:
        data, _ = compiler.compile_runtime(base.root_ir(kernel.source), exit=exit_name)
    source = json.loads(data)["source"]
    if exit_name == "bangc":
        assert "for(long long z=0;z<144;z++)rec[z]=base[z*" + str(stride) in source
    else:
        for scale in ("ax=tl.load(", "bx=tl.load("):
            address = source.split(scale, 1)[1].split(",other=", 1)[0]
            assert "tl.cast(g,tl.int64)*" + str(stride) in address
