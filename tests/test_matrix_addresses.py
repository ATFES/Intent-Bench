"""Compile valid strided views whose last offsets exceed signed i32."""

import json
import os
from contextlib import ExitStack

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


def test_large_scale_stride_on_cuda():
    """Map only the three scale pages read by a real FP8 slow-path call."""
    if not os.environ.get("INTENTDSL_LIBRARY"):
        pytest.skip("requires an installed IntentDSL compiler")
    torch = pytest.importorskip("torch")
    cuda = pytest.importorskip("cuda.bindings.driver")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("requires an FP8-capable CUDA device")
    import intentdsl
    from intentdsl import examples

    def checked(result):
        status, *values = result
        assert status == cuda.CUresult.CUDA_SUCCESS, status
        return values[0] if values else None

    device = torch.device("cuda", torch.cuda.current_device())
    a = torch.ones((1, 3, 32), device=device).to(torch.float8_e4m3fn)
    b = a.reshape(3, 32, 1)
    prop = cuda.CUmemAllocationProp()
    prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device.index
    access = cuda.CUmemAccessDesc()
    access.location = prop.location
    access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    page = checked(cuda.cuMemGetAllocationGranularity(
        prop, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
    stride = 1 << 30
    with ExitStack() as cleanup:
        base = int(checked(cuda.cuMemAddressReserve(2 * stride + page, 0, 0, 0)))
        cleanup.callback(cuda.cuMemAddressFree, base, 2 * stride + page)
        for group in range(3):
            address = base + group * stride
            memory = checked(cuda.cuMemCreate(page, prop, 0))
            cleanup.callback(cuda.cuMemRelease, memory)
            checked(cuda.cuMemMap(address, page, 0, memory, 0))
            cleanup.callback(cuda.cuMemUnmap, address, page)
            checked(cuda.cuMemSetAccess(address, page, [access], 1))
            checked(cuda.cuMemsetD8(address, 127 + group, 1))

        class ScalePages:
            __cuda_array_interface__ = dict(shape=(3,), strides=(stride,), typestr="|u1",
                                            data=(base, False), version=2)

        scale = torch.as_tensor(ScalePages(), device=device)
        cleanup.callback(torch.cuda.current_stream(device).synchronize)
        assert scale.data_ptr() == base and scale.stride() == (stride,)
        result = intentdsl.jit(examples.fp8_gemm, backend="triton").run(
            a, scale[None, :], b, scale[:, None])
        assert result.item() == 32 * (1 + 4 + 16)
