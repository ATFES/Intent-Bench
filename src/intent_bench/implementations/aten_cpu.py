"""Prepared ATen CPU calls through libtorch, timed without Python submission."""
import ctypes as C
from functools import cache
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

from ..model import PreparedCall
from ..output import output_root


OPERATIONS = {name: index for index, name in enumerate((
    "relu", "softmax", "transpose", "embedding", "layer_norm", "gemm", "batched_gemm",
    "swiglu", "cumsum", "adamw", "batch_norm", "cholesky", "group_norm_backward",
    "csr_spmv", "ragged_gemm", "int8_gemm", "attention", "histogram", "jagged_mean",
    "batched_gemm_tn", "batched_gemm_nt", "batched_gemm_tt", "causal_convolution", "nonzero",
    "fp8_gemm", "paged_decode", "mamba_chunk_state"))}


@cache
def library(root, compiler, onednn=False, dnnl_prefix=None):
    import torch
    from torch.utils.cpp_extension import include_paths, library_paths
    parent = root/"build/native"
    parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix="aten-cpu-", dir=parent))
    source = Path(__file__).parents[1]/"native/aten_cpu.cpp"
    command = [compiler, "-std=c++17", "-O3", "-shared", "-fPIC",
               "-D_GLIBCXX_USE_CXX11_ABI="+str(int(torch.compiled_with_cxx11_abi())), str(source),
               "-o", str(build/"aten.so")]
    command += ["-I"+path for path in include_paths()]
    command += [option for path in library_paths() for option in ("-L"+path, "-Wl,-rpath,"+path)]
    command += ["-ltorch_cpu", "-lc10"]
    if onednn:
        # Compile against the system oneDNN headers separately from libtorch's
        # bundled private headers. The bridge exposes only its own plain C ABI.
        matrix_source=source.with_name("onednn_matmul.cpp")
        matrix_object=build/"onednn_matmul.o"
        include=["-I"+str(Path(dnnl_prefix)/"include")] if dnnl_prefix else []
        result=subprocess.run([compiler,"-std=c++17","-O3","-fPIC",*include,"-c",str(matrix_source),
                               "-o",str(matrix_object)],text=True,capture_output=True,timeout=180)
        (build/"onednn-compile.log").write_text(result.stdout+result.stderr)
        if result.returncode:
            from ..corpus._common.measurement import PipelineStageError
            raise PipelineStageError("source_native_compilation",
                f"oneDNN 3.x development headers are required (environment option dnnl_prefix); "
                f"see {build/'onednn-compile.log'}\n{result.stderr}")
        if dnnl_prefix:
            directory=str(Path(dnnl_prefix)/"lib")
            command += ["-L"+directory,"-Wl,-rpath,"+directory]
        command += ["-DINTENT_BENCH_ONEDNN",str(matrix_object),"-ldnnl"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=180)
    (build/"compile.log").write_text(result.stdout+result.stderr)
    if result.returncode:
        from ..corpus._common.measurement import PipelineStageError
        raise PipelineStageError("source_native_compilation", f"ATen binding build failed: {build/'compile.log'}\n{result.stderr}")
    native = C.CDLL(str(build/"aten.so"))
    native.bench_aten_create.argtypes = [C.c_int, C.c_int, C.c_int, C.POINTER(C.c_void_p),
        C.POINTER(C.c_int), C.POINTER(C.c_int64), C.POINTER(C.c_int64), C.POINTER(C.c_int64),
        C.c_int, C.POINTER(C.c_double)]
    native.bench_aten_create.restype = C.c_void_p
    native.bench_aten_run.argtypes = [C.c_void_p]
    native.bench_aten_run.restype = None
    native.bench_aten_error.argtypes = [C.c_void_p]
    native.bench_aten_error.restype = C.c_char_p
    native.bench_aten_implementation.argtypes = [C.c_void_p]
    native.bench_aten_implementation.restype = C.c_char_p
    native.bench_aten_outputs.argtypes = [C.c_void_p]
    native.bench_aten_outputs.restype = C.c_int
    native.bench_aten_output.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int64), C.POINTER(C.c_int64), C.POINTER(C.c_int)]
    native.bench_aten_output.restype = C.c_void_p
    native.bench_aten_destroy.argtypes = [C.c_void_p]
    native.bench_aten_destroy.restype = None
    return native, build


def prepare(name, arguments, environment):
    import torch
    from .native_sampling import sample_timer
    from .native_sequence import NativeSequence
    if environment.device != "cpu":
        raise ValueError("ATen native CPU comparisons require device=cpu")
    tensors = [value for value in arguments if isinstance(value, torch.Tensor)]
    scalars = [value for value in arguments if not isinstance(value, torch.Tensor)]
    dtypes = (torch.float32, torch.float16, torch.bfloat16, torch.int64, torch.int32, torch.int8,
              torch.uint8, torch.float8_e4m3fn)
    if any(value.device.type != "cpu" or not value.is_contiguous() for value in tensors):
        raise ValueError("this ATen native binding requires contiguous CPU inputs")
    if name == "adamw" and (scalars[3] != 1-scalars[1] or scalars[4] != 1-scalars[2]):
        raise ValueError("the standard AdamW case requires first-step bias correction")
    native, build = library(output_root(environment.output_root), environment.options.get("cxx", "c++"),
                            name == "mamba_chunk_state", environment.options.get("dnnl_prefix"))
    dimensions = [size for tensor in tensors for size in tensor.shape]
    strides = [stride for tensor in tensors for stride in tensor.stride()]
    state = C.c_void_p(native.bench_aten_create(OPERATIONS[name], environment.threads, len(tensors),
        (C.c_void_p*len(tensors))(*(tensor.data_ptr() for tensor in tensors)),
        (C.c_int*len(tensors))(*(dtypes.index(tensor.dtype) for tensor in tensors)),
        (C.c_int64*len(tensors))(*(tensor.ndim for tensor in tensors)),
        (C.c_int64*len(dimensions))(*dimensions), (C.c_int64*len(strides))(*strides),
        len(scalars), (C.c_double*len(scalars))(*scalars)))

    def checked():
        if not state.value:
            raise RuntimeError("ATen native call has been closed")
        error = native.bench_aten_error(state)
        if error:
            raise RuntimeError(error.decode())

    class Call:
        def native_call(self):
            checked()
            return SimpleNamespace(function=native.bench_aten_run, argument_types=(C.c_void_p,),
                                   arguments=(state,), owners=(native, tensors), validate=checked)
        def launch(self):
            checked()
            native.bench_aten_run(state)
            checked()
        def result(self):
            checked()
            outputs = []
            for index in range(native.bench_aten_outputs(state)):
                rank, shape, kind = C.c_int64(), (C.c_int64*16)(), C.c_int()
                pointer = native.bench_aten_output(state, index, C.byref(rank), shape, C.byref(kind))
                sizes = tuple(shape[:rank.value])
                from math import prod
                count = prod(sizes)
                width = torch.empty((), dtype=dtypes[kind.value]).element_size()
                storage = (C.c_ubyte*(count*width)).from_address(pointer)
                outputs.append(torch.frombuffer(storage, dtype=dtypes[kind.value]).reshape(sizes).clone())
            return outputs[0] if len(outputs) == 1 else tuple(outputs)
        def close(self):
            if state.value:
                native.bench_aten_destroy(state)
                state.value = None

    call = Call()
    try:
        checked()
    except Exception:
        call.close()
        raise
    metadata = {"library": "PyTorch libtorch/ATen CPU", "native_build": str(build),
                "scope_note": "Prepared native ATen operations; output/workspace allocation and Python submission excluded."}
    if name == "ragged_gemm":
        metadata["scope_note"] += " Reading group offsets, slicing views and every group GEMM are inside the native interval."
    if name == "int8_gemm":
        metadata["library"] = "ATen CPU INT8 GEMM with INT32 bias"
        metadata["scope_note"] += " Integer GEMM and bias addition are both timed."
    if name == "attention":
        metadata["library"] = "ATen CPU Flash Attention, causal grouped queries"
        metadata["scope_note"] = ("Public native CPU flash-attention operator on original grouped Q/K/V; "
                                  "packing, output/LSE allocation and auxiliary LSE are included. Python submission is excluded.")
    if name == "paged_decode":
        metadata["library"] = "ATen page gathering and CPU Flash Attention"
        metadata["scope_note"] = ("Original FP16 paged KV and grouped queries; runtime page-table reads, page gathering, "
                                  "layout preparation, CPU flash attention and output writeback are all timed. "
                                  "This is a native library composition, not a dedicated paged-attention kernel. "
                                  "The library's packed KV and auxiliary LSE allocations are included.")
    if name == "mamba_chunk_state":
        metadata["library"] = "ATen chunk scaling and oneDNN BF16 matmul with FP32 output"
        metadata["matmul_implementation"] = native.bench_aten_implementation(state).decode()
        metadata["scope_note"] = ("Complete native library composition: FP32 decay/scale, grouped basis expansion, "
                                  "BF16 weighted-basis rounding, X transpose and BF16 x BF16 -> FP32 oneDNN matmul. "
                                  "All data preparation is timed; primitive construction and reusable allocation are outside.")
    if name == "histogram":
        metadata["library"] = "ATen CPU bincount"
        metadata["scope_note"] = ("Complete native signed-index conversion, excluded-bin mapping, bincount and INT32 output. "
                                  "The library's count allocation is included; external workspace and Python submission are excluded.")
    if name == "jagged_mean":
        metadata["library"] = "ATen CPU segment_reduce(mean)"
        metadata["scope_note"] += " Segment offsets are read by the native library inside every interval."
    if name.startswith("batched_gemm"):
        metadata["library"] = "ATen CPU batched GEMM"
        metadata["scope_note"] += " Transposed views and any library packing are inside the native interval; input dtype is unchanged."
    if name == "causal_convolution":
        metadata["library"] = "ATen CPU grouped conv1d and SiLU"
        metadata["scope_note"] = ("FP16 inputs widened to FP32 to preserve source products and accumulation; "
                                  "conversion, grouped convolution, causal slicing, SiLU and FP16 output all timed. "
                                  "The native library's convolution allocation is included.")
    if name == "nonzero":
        metadata["library"] = "ATen CPU nonzero and bincount"
        metadata["scope_note"] = ("Native stable coordinate extraction, row counts, INT32 index conversion and "
                                  "padding to the author's fixed row capacity all timed, including library temporary allocations.")
    if name == "fp8_gemm":
        metadata["library"] = "ATen CPU E4M3/E8M0 decoding and FP32 GEMM"
        metadata["scope_note"] = ("Original E4M3 inputs and E8M0 scale bytes; decoding, scale application and native FP32 GEMM "
                                  "all timed. Reusable output/workspace allocation is outside the interval. "
                                  "This is a library composition, not a fused native FP8 matrix instruction.")
    prepared = PreparedCall(call.launch, call.result, close=call.close, owners=call,
        device="cpu", timing="cpu_native_monotonic", metadata=metadata)
    def measure(settings):
        timer = NativeSequence((call,))
        def interval():
            elapsed = timer.run()
            # C++ exceptions are retained across the C ABI. Surface them after
            # every interval, including the final sample, outside its timing.
            checked()
            return elapsed
        try:
            return sample_timer(SimpleNamespace(run=interval), settings,
                                prepared.reset or (lambda: None), metadata=metadata)
        finally:
            timer.close()
    prepared.native_measure = measure
    return prepared
