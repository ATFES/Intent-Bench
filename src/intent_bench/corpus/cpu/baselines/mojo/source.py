from pathlib import Path
from dataclasses import dataclass

import ctypes
import os
import shutil
import statistics
import subprocess
import tempfile

_libraries = {}


@dataclass(frozen=True)
class SourceParameter:
    """A hand-written corpus entry's declared public parameter and C ABI slots."""

    declaration: dict
    slots: tuple[dict, ...]


def view(name: str, dimensions: tuple[int, ...], *, output: bool = False,
         dtype: str = "f32", compact: bool = False) -> SourceParameter:
    return SourceParameter(
        {"name": name, "kind": "view", "dtype": dtype, "access": 1 if output else 0,
         "shape": [-1] * len(dimensions), "dimensions": list(dimensions),
         "strides": [None] * len(dimensions), "alias": "", "noalias": False},
        ({"role": "pointer", "carrier": "ptr", "element": dtype},) if compact else
        ({"role": "pointer", "carrier": "ptr", "element": dtype},
         *({"role": "extent", "axis": axis, "carrier": "i64"} for axis in range(len(dimensions))),
         *({"role": "stride", "axis": axis, "carrier": "i64"} for axis in range(len(dimensions)))),
    )


def scalar(name: str, dtype: str = "f32") -> SourceParameter:
    return SourceParameter({"name": name, "kind": "scalar", "dtype": dtype},
                           ({"role": "scalar", "carrier": dtype},))


def prepare_source(path: Path, entry: str, parameters, target, arguments, *, required_workers=8):
    if required_workers is not None and target.workers != required_workers:
        raise NotImplementedError("the initial Mojo source corpus uses an explicit eight-worker budget")
    import torch
    dtypes={"f32":torch.float32,"f16":torch.float16,"bf16":torch.bfloat16,"i64":torch.int64,
            "i32":torch.int32,"u8":torch.uint8,"f8e4m3fn":torch.float8_e4m3fn}
    executable = getattr(target, "executable", None) or os.environ.get("INTENT_BENCH_MOJO") or os.environ.get("INTENT_MOJO") or shutil.which("mojo")
    if not executable:
        raise RuntimeError("select a Mojo executable with INTENT_BENCH_MOJO")
    types = {"ptr": ("void*", ctypes.c_void_p), "i64": ("int64_t", ctypes.c_int64), "f32": ("float", ctypes.c_float)}
    slots = [types[slot["carrier"]] for parameter in parameters for slot in parameter.slots]
    key = (str(path), entry, str(executable))
    if key not in _libraries:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home()/".cache"))/"intent-bench/native"
        root.mkdir(parents=True, exist_ok=True)
        build = Path(tempfile.mkdtemp(prefix="source-", dir=root))
        command = [str(executable), "build", "--emit", "shared-lib", str(path), "-o", str(build/"source.so")]
        result = subprocess.run(command, text=True, capture_output=True)
        (build/"build.log").write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(f"Mojo source build failed; {build/'build.log'}: {result.stderr}")
        signature = ",".join(c for c, _ in slots)
        call_args = ",".join(f"*({c}*)args[{i}]" for i, (c, _) in enumerate(slots))
        shim = '#include <stdint.h>\n#include <time.h>\n'
        shim += '#include <xmmintrin.h>\nuint32_t bench_native_enter_ieee(void){uint32_t s=_mm_getcsr();_mm_setcsr(s & ~0x8040u);return s;}\nvoid bench_native_leave_ieee(uint32_t s){_mm_setcsr(s);}\n'
        shim += f'typedef void (*entry)({signature});\n'
        shim += 'double measure(void *function, void **args, int64_t count) { struct timespec a,b; clock_gettime(CLOCK_MONOTONIC,&a);'
        shim += f'for(int64_t i=0;i<count;++i)((entry)function)({call_args});'
        shim += 'clock_gettime(CLOCK_MONOTONIC,&b);return ((b.tv_sec-a.tv_sec)*1000.+(b.tv_nsec-a.tv_nsec)/1e6)/count;}\n'
        (build/"timer.c").write_text(shim)
        subprocess.run(["cc","-O3","-shared","-fPIC",str(build/"timer.c"),"-o",str(build/"timer.so")],check=True)
        timer = ctypes.CDLL(str(build/"timer.so"), mode=ctypes.RTLD_GLOBAL)
        library = ctypes.CDLL(str(build/"source.so"))
        function = getattr(library,entry); function.argtypes=[t for _, t in slots]; function.restype=None
        timer.measure.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_void_p),ctypes.c_int64]
        timer.measure.restype=ctypes.c_double
        _libraries[key]=(library,function,timer,build)
    library,function,timer,build=_libraries[key]
    inputs=iter(arguments); dimensions={}; values=[]; outputs=[]
    for parameter in parameters:
        declaration=parameter.declaration
        if declaration["kind"]=="view" and declaration["access"]==1:
            value=torch.empty(tuple(dimensions[d] for d in declaration["dimensions"]),dtype=dtypes[declaration["dtype"]])
            outputs.append(value)
        else:
            value=next(inputs)
            if declaration["kind"]=="view":
                if value.dtype!=dtypes[declaration["dtype"]] or value.device.type!="cpu" or not value.is_contiguous():
                    raise ValueError(f"native Mojo source requires contiguous CPU {declaration['dtype']} input")
                for identity,size in zip(declaration["dimensions"],value.shape,strict=True):
                    if identity in dimensions and dimensions[identity]!=size:raise ValueError("conflicting input dimensions")
                    dimensions[identity]=size
        values.append(value)
    bound=[]
    for parameter,value in zip(parameters,values,strict=True):
        for slot in parameter.slots:
            raw=value.data_ptr() if slot["role"]=="pointer" else value.shape[slot["axis"]] if slot["role"]=="extent" else value.stride(slot["axis"]) if slot["role"]=="stride" else value
            carrier = types[slot["carrier"]][1]
            bound.append(raw if isinstance(raw, carrier) else carrier(raw))
    addresses=(ctypes.c_void_p*len(bound))(*(ctypes.addressof(v) for v in bound))
    class Call:
        def __init__(self):
            self.owners=values
            self.library=library
        def launch(self):
            function(*bound)
        def run(self):
            return timer.measure(function,addresses,1)
        def result(self):return outputs[0] if len(outputs)==1 else tuple(outputs)
        def samples(self,count=7,warmup=3):
            for _ in range(warmup):self.launch()
            pilot=timer.measure(function,addresses,1)
            repeat=max(1,min(1000,int(2/max(pilot,1e-6))))
            return [timer.measure(function,addresses,repeat) for _ in range(count)]
        def benchmark(self):return statistics.median(self.samples())
    return Call()
