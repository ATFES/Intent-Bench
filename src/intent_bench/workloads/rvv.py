import numpy as np
from ..model import Case,Implementation,Workload

NAMES=("gemm","q4_projection","rms_norm","q4_dot","q8_quantize")


def q8_reference(case,inputs):
    x=inputs[1].reshape(-1,256)
    records=[]
    for row in x:
        maximum=row[np.abs(row).argmax()]
        scale=np.float32(-127)/maximum if maximum else np.float32(0)
        values=np.minimum(127,np.rint(row*scale)).astype("int8")
        sums=values.reshape(16,16).sum(1).astype("<i2")
        d=np.array([np.float32(1)/scale if maximum else 0],dtype="<f4")
        records.append(d.tobytes()+values.tobytes()+sums.tobytes())
    return np.frombuffer(b"".join(records),dtype="uint8").copy()


def build(name):
    if name not in NAMES:raise KeyError(name)
    cases=[Case("decode",{"m":1,"n":128,"k":1024}),Case("prefill",{"m":128,"n":128,"k":1024})]
    if name=="rms_norm":cases=[Case("rows",{"m":32,"n":1024,"k":1024})]
    if name in {"q4_dot","q8_quantize"}:cases=[Case("vector",{"m":1,"n":1,"k":4096})]
    def inputs(case,seed):
        rng=np.random.default_rng(seed);p=case.parameters
        return (rng.normal(size=(p["n"],p["k"])).astype("float32"),rng.normal(size=(p["m"],p["k"])).astype("float32"))
    def prepare(case,data,env,reference=False):
        from ..implementations.native_libraries import ggml
        return ggml(name,case,data,env,reference=reference)
    if name=="q8_quantize":oracle=q8_reference
    elif name=="rms_norm":oracle=lambda case,x:x[1]/np.sqrt((x[1]**2).mean(-1,keepdims=True)+1e-5)
    elif name=="gemm":oracle=lambda case,x:x[1].astype("float64")@x[0].astype("float64").T
    else:oracle=Implementation("dequantized_scalar_oracle",lambda case,data,env:prepare(case,data,env,True))
    def generated(case,data,env):
        from ..implementations.weft_intent import prepare
        if name=='gemm':
            return prepare('intent.examples.kernels.contraction.gemm:gemm_f32',(data[1],data[0].T.copy()),env)
        if name=='rms_norm':
            return prepare('intent.examples.kernels.normalization.rms_norm:rms_norm_f32',(data[1],1/data[1].shape[-1],1e-5),env)
        prefix='intent.examples.kernels.quantization.quantized_projection:'
        if name=='q8_quantize':
            call=prepare(prefix+name,(data[1].reshape(-1,256),),env)
            result=call.result;call.result=lambda:result().reshape(-1)
            return call
        from ..implementations.native_libraries import pack_q4
        weights=pack_q4(data,env)
        if name=='q4_projection':args=(weights,data[1].reshape(data[1].shape[0],-1,256))
        else:args=(weights[0],q8_reference(case,data).reshape(-1,292))
        return prepare(prefix+name,args,env)
    return Workload("rvv/"+name,cases,inputs,{"ggml":Implementation("ggml",prepare),
                    "intent:weft":Implementation("intent:weft",generated)},oracle=oracle,
                    atol=2e-3,rtol=2e-3,family="rvv/"+name)
