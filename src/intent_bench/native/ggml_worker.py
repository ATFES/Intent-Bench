import base64
import ctypes as C
from pathlib import Path
import subprocess
import time
import numpy as np
from cnnl_worker import serve


class Worker:
    def __init__(self,request):
        root=Path(__file__).parent;source=Path(request["source"]);libraries=Path(request["libraries"])
        start=time.perf_counter()
        build=subprocess.run([request.get("cxx","c++"),"-O3","-std=c++17","-shared","-fPIC",str(root/"ggml_ops.cpp"),
            "-I"+str(source/"ggml/include"),"-L"+str(libraries),"-Wl,-rpath,"+str(libraries),
            "-lggml","-lggml-cpu","-lggml-base","-o",str(root/"ops.so")],text=True,capture_output=True)
        if build.returncode:raise RuntimeError(build.stdout+build.stderr)
        self.compile_seconds=time.perf_counter()-start
        self.lib=C.CDLL(str(root/"ops.so"))
        self.lib.bench_create.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int]
        self.lib.bench_create.restype=C.c_void_p
        self.lib.bench_run.argtypes=[C.c_void_p];self.lib.bench_run.restype=C.c_int
        self.lib.bench_output.argtypes=[C.c_void_p,C.c_void_p,C.c_int];self.lib.bench_output.restype=None
        self.lib.bench_measure.argtypes=[C.c_void_p,C.c_int];self.lib.bench_measure.restype=C.c_double
        self.lib.bench_destroy.argtypes=[C.c_void_p];self.lib.bench_destroy.restype=None
        self.lib.bench_error.argtypes=[];self.lib.bench_error.restype=C.c_char_p
        self.lib.bench_packed_weight.argtypes=[C.c_void_p,C.c_void_p];self.lib.bench_packed_weight.restype=None
        self.lib.bench_gated_delta_create.argtypes=[C.POINTER(C.c_void_p)]+[C.c_int]*6
        self.lib.bench_gated_delta_create.restype=C.c_void_p
        self.lib.bench_state_output.argtypes=[C.c_void_p,C.c_void_p];self.lib.bench_state_output.restype=None
        if request["operation"] == "gated_delta":
            self.kind=6
            if request.get("reference",False):
                raise ValueError("GGML gated delta is a library baseline, not a numerical oracle")
            self.inputs=[np.load(root/f"input{i}.npy") for i in range(5)]
            q,k,v,gate,beta=self.inputs
            if any(value.dtype!=np.uint16 or not value.flags.c_contiguous for value in self.inputs):
                raise ValueError("GGML gated delta binding requires contiguous BF16 storage bits")
            if q.ndim!=4 or k.shape!=q.shape or v.ndim!=4:
                raise ValueError("GGML gated delta requires Q/K/V in [B,T,H,D] order")
            batch,tokens,key_heads,dimension=q.shape
            value_heads=v.shape[2]
            if (v.shape!=(batch,tokens,value_heads,dimension) or value_heads%key_heads or
                    gate.shape!=(batch,tokens,value_heads) or beta.shape!=gate.shape):
                raise ValueError("GGML gated delta requires equal key/value dimensions and grouped heads")
            pointers=(C.c_void_p*5)(*(value.ctypes.data for value in self.inputs))
            self.context=self.lib.bench_gated_delta_create(pointers,batch,tokens,key_heads,value_heads,
                                                         dimension,request["threads"])
            if not self.context:raise RuntimeError(self.lib.bench_error().decode())
            self.output=np.empty(v.shape,dtype=np.uint16)
            self.state=np.empty((batch,value_heads,dimension,dimension),dtype=np.float32)
            return
        self.a=np.load(root/"input0.npy");self.b=np.load(root/"input1.npy")
        self.kind={"gemm":0,"q4_projection":1,"rms_norm":2,"q4_dot":3,"q8_quantize":4,
                   "q4_projection_packed":5}[request["operation"]]
        if self.kind==5:
            if self.a.dtype!=np.uint8 or self.a.ndim!=3 or self.a.shape[-1]!=144 or self.b.dtype!=np.float32:
                raise ValueError('packed Q4_K projection requires [N,G,144] bytes and FP32 activation')
            n,groups,_=self.a.shape
            if self.b.shape!=(groups,256):raise ValueError('activation must match the Q4_K groups')
            self.b=self.b.reshape(1,-1)
        m,k=self.b.shape;n=self.a.shape[0]
        self.context=self.lib.bench_create(self.kind,self.a.ctypes.data,self.b.ctypes.data,m,n,k,request["threads"])
        if not self.context:raise RuntimeError(self.lib.bench_error().decode())
        self.output=np.empty((m,n),dtype="float32") if self.kind!=4 else np.empty(m*k//256*292,dtype="uint8")
        self.reference=request.get("reference",False)

    def run(self):
        if self.lib.bench_run(self.context):raise RuntimeError(self.lib.bench_error().decode())

    def result(self):
        if self.kind==6:
            self.lib.bench_output(self.context,self.output.ctypes.data,0)
            self.lib.bench_state_output(self.context,self.state.ctypes.data)
            return {"items":[
                {"shape":list(self.output.shape),"torch_dtype":"bfloat16",
                 "data":base64.b64encode(self.output.tobytes()).decode()},
                {"shape":list(self.state.shape),"dtype":"float32",
                 "data":base64.b64encode(self.state.tobytes()).decode()}]}
        self.lib.bench_output(self.context,self.output.ctypes.data,int(self.reference))
        return {"shape":list(self.output.shape),"dtype":str(self.output.dtype),"data":base64.b64encode(self.output.tobytes()).decode()}

    def packed(self):
        if self.kind not in (1,3):raise ValueError('only Q4_K weights have this packed-input contract')
        weights=np.empty((self.a.shape[0],self.a.shape[1]//256,144),dtype='uint8')
        self.lib.bench_packed_weight(self.context,weights.ctypes.data)
        return {'shape':list(weights.shape),'dtype':'uint8','data':base64.b64encode(weights.tobytes()).decode()}

    def measure(self,samples,warmup,cache):
        for _ in range(warmup):self.run()
        eviction=np.empty(64*1024*1024,dtype=np.uint8) if cache=='flush' else None
        values=[]
        for _ in range(samples):
            if eviction is not None:eviction.fill(0)
            elapsed=self.lib.bench_measure(self.context,1)
            if elapsed<=0:raise RuntimeError(self.lib.bench_error().decode())
            values.append(elapsed)
        return values

    def close(self):self.lib.bench_destroy(self.context)


if __name__=="__main__":serve(Worker)
