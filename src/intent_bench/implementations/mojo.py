"""Raw Mojo C ABI baseline, independent of Intent's compiler and runtime."""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from ..model import EnvironmentUnavailable, PreparedCall

_libraries={}
NAMES=("relu","swiglu","transpose","embedding","jagged_mean","prefix_sum","csr_spmv","histogram","adamw")


def library(compiler=None):
    compiler=compiler or os.environ.get("INTENT_BENCH_MOJO") or shutil.which("mojo")
    if compiler is None:
        raise EnvironmentUnavailable("set INTENT_BENCH_MOJO or install Mojo")
    if compiler not in _libraries:
        root=Path(os.environ.get("XDG_CACHE_HOME",Path.home()/".cache"))/"intent-bench/native"
        root.mkdir(parents=True,exist_ok=True)
        build=Path(tempfile.mkdtemp(prefix="mojo-",dir=root))
        source=Path(__file__).parents[1]/"native"
        for command in ([str(compiler),"build","--emit","shared-lib",str(source/"ops.mojo"),"-o",str(build/"ops.so")],
                        ["cc","-O3","-shared","-fPIC",str(source/"timer.c"),"-o",str(build/"timer.so")]):
            result=subprocess.run(command,text=True,capture_output=True)
            (build/("mojo.log" if command[0]==str(compiler) else "timer.log")).write_text(result.stdout+result.stderr)
            if result.returncode:
                raise RuntimeError(f"native baseline compilation failed: {result.stderr}; artifacts: {build}")
        native=ctypes.CDLL(str(build/"ops.so"));timer=ctypes.CDLL(str(build/"timer.so"))
        native.bench_run.argtypes=[ctypes.c_int64]+[ctypes.c_void_p]*9+[ctypes.c_int64]*3
        native.bench_run.restype=None
        timer.bench_time.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(ctypes.c_int64),ctypes.c_int64]
        timer.bench_time.restype=ctypes.c_double
        _libraries[compiler]=(native,timer,build)
    return _libraries[compiler]


def prepare(name, case, inputs, env):
    native,timer,build=library(env.options.get("mojo"))
    op=NAMES.index(name)
    p=case.parameters;m,n=p["rows"],p["columns"]
    a=inputs.get("a",np.zeros(1,dtype="float32"))
    outputs=[np.empty_like(a),np.empty_like(a),np.empty_like(a)]
    if name=="transpose":outputs[0]=np.empty((n,m),dtype="float32")
    elif name=="embedding":outputs[0]=np.empty((len(inputs["indices"]),n),dtype="float32")
    elif name=="jagged_mean":outputs[0]=np.empty((m,n),dtype="float32")
    elif name=="csr_spmv":outputs[0]=np.empty(m,dtype="float32")
    elif name=="histogram":outputs[0]=np.empty(n,dtype="float32")
    arrays=[inputs.get(k,np.zeros(1,dtype="float32")) for k in ("a","b","c","d")]+outputs
    arrays += [inputs.get(k,np.zeros(1,dtype="int64")) for k in ("indices","offsets")]
    if any(not x.flags.c_contiguous for x in arrays):
        raise ValueError("this native Mojo implementation requires contiguous inputs")
    ptrs=(ctypes.c_void_p*9)(*(x.ctypes.data for x in arrays))
    count=len(inputs["indices"]) if name in {"embedding","histogram"} else a.size
    shapes=(ctypes.c_int64*3)(m,n,count)
    def run():
        # Keep every allocation alive with the prepared call, including unused ABI slots.
        native.bench_run(op,*(x.ctypes.data for x in arrays),*shapes)
    def result():return tuple(outputs) if name=="adamw" else outputs[0]
    metadata = {"artifact": str(build), "baseline": "independent Mojo", "workers": 1}
    def measure(settings):
        from .native_sampling import sample_native_call
        return sample_native_call(
            lambda: timer.bench_time(native.bench_run, op, ptrs, shapes, 1),
            settings, lambda: None, metadata=metadata,
        )
    return PreparedCall(run,result,native_measure=measure,timing="cpu_native_monotonic",
                        metadata=metadata)
