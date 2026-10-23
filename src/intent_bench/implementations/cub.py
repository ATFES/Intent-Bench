"""CUB device primitives; binding and allocation stay outside operator timing."""
import ctypes
from functools import cache
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from ..model import EnvironmentUnavailable, PreparedCall
from ..output import output_root


@cache
def library(compiler, architecture, root):
    from ..corpus._common.measurement import PipelineStageError
    if compiler is None:
        compiler = shutil.which("nvcc")
        if compiler is None:
            compiler = str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))/"bin/nvcc")
    compiler = shutil.which(compiler)
    if compiler is None:
        raise EnvironmentUnavailable("CUB requires nvcc; set the environment nvcc option or CUDA_HOME")
    parent = root/"build/native"
    parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix=f"cub-sm{architecture}-", dir=parent))
    source = Path(__file__).parents[1]/"native/cub_histogram.cu"
    command = [compiler, "-std=c++17", "-O3", "--shared", "-Xcompiler", "-fPIC",
               f"-arch=sm_{architecture}", str(source), "-o", str(build/"cub.so")]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired as error:
        def decoded(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        (build/"compile.log").write_text(decoded(error.stdout)+decoded(error.stderr))
        raise PipelineStageError("source_native_compilation",
            f"CUB compilation exceeded 180 seconds; see {build/'compile.log'}") from error
    (build/"compile.log").write_text(completed.stdout+completed.stderr)
    if completed.returncode:
        raise PipelineStageError("source_native_compilation",
            f"CUB compilation failed; see {build/'compile.log'}\n{completed.stderr}")
    native = ctypes.CDLL(str(build/"cub.so"))
    native.bench_cub_histogram.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p]
    native.bench_cub_histogram.restype = ctypes.c_int
    native.bench_cuda_error.argtypes = [ctypes.c_int]
    native.bench_cuda_error.restype = ctypes.c_char_p
    native.bench_cub_version.restype = ctypes.c_int
    return native, build


def histogram(samples, environment):
    import torch
    from ..corpus._common.measurement import report_stage
    if samples.device.type != "cuda" or samples.dtype != torch.float32 or not samples.is_contiguous():
        raise ValueError("CUB histogram requires contiguous CUDA float32 samples")
    major, minor = torch.cuda.get_device_capability(samples.device)
    report_stage("source_native_compilation")
    native, build = library(environment.options.get("nvcc"), f"{major}{minor}",
                            output_root(environment.output_root))
    report_stage("adapter_preparation")
    output = torch.empty(256, dtype=torch.int32, device=samples.device)
    size = ctypes.c_size_t()

    def invoke(temporary):
        status = native.bench_cub_histogram(temporary, ctypes.byref(size), samples.data_ptr(),
            output.data_ptr(), samples.numel(), torch.cuda.current_stream(samples.device).cuda_stream)
        if status:
            raise RuntimeError(f"CUB histogram failed: {native.bench_cuda_error(status).decode()}")

    with torch.cuda.device(samples.device):
        invoke(None)
        workspace = torch.empty(size.value, dtype=torch.uint8, device=samples.device)

    def launch():
        with torch.cuda.device(samples.device):
            invoke(workspace.data_ptr())

    return PreparedCall(launch, lambda: output, device=str(samples.device),
        metadata={"library": "CUB DeviceHistogram::HistogramEven",
                  "cub_version": native.bench_cub_version(), "native_build": str(build),
                  "workspace_bytes": size.value,
                  "scope_note": "Complete CUB histogram, including output initialization and merge; f32-to-i32 input conversion is fused into its iterator."},
        owners=(samples, output, workspace, native))
