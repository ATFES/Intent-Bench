import numpy as np

from ..model import Case,Implementation,PreparedCall,Workload
from ..implementations.mojo import NAMES


def inputs(name,case,seed):
    rng=np.random.default_rng(seed);m,n=case.parameters["rows"],case.parameters["columns"]
    data={key:rng.normal(size=(m,n)).astype("float32") for key in ("a","b","c")}
    data["d"]=np.abs(rng.normal(size=(m,n))).astype("float32")
    if name=="embedding":data["indices"]=rng.integers(0,m,size=m//2+1,dtype="int64")
    if name=="histogram":data["indices"]=rng.integers(-2,n+2,size=m*n,dtype="int64")
    if name=="jagged_mean":
        data["a"]=rng.normal(size=(m*3,n)).astype("float32")
        data["offsets"]=np.arange(m+1,dtype="int64")*3
    if name=="csr_spmv":
        data["a"]=rng.normal(size=m*4).astype("float32")
        data["b"]=rng.normal(size=n).astype("float32")
        data["indices"]=rng.integers(0,n,size=m*4,dtype="int64")
        data["offsets"]=np.arange(m+1,dtype="int64")*4
    return data


def reference(name,case,x):
    a=x["a"];m,n=case.parameters["rows"],case.parameters["columns"]
    if name=="relu":return np.maximum(a,0)
    if name=="swiglu":return a/(1+np.exp(-a))*x["b"]
    if name=="transpose":return a.T.copy()
    if name=="embedding":return a[x["indices"]]
    if name=="prefix_sum":return np.cumsum(a,axis=-1,dtype="float32")
    if name=="jagged_mean":return np.stack([a[x["offsets"][i]:x["offsets"][i+1]].mean(0) for i in range(m)])
    if name=="csr_spmv":return np.array([np.dot(a[x["offsets"][i]:x["offsets"][i+1]],x["b"][x["indices"][x["offsets"][i]:x["offsets"][i+1]]]) for i in range(m)],dtype="float32")
    if name=="histogram":
        ids=x["indices"];return np.bincount(ids[(ids>=0)&(ids<n)],minlength=n).astype("float32")
    if name=="adamw":
        first=np.float32(.9)*x["c"]+np.float32(.1)*x["b"]
        second=np.float32(.999)*x["d"]+np.float32(.001)*x["b"]**2
        output=a*np.float32(1-.001*.01)-np.float32(.001)*(first/np.float32(.1))/(np.sqrt(second/np.float32(.001))+np.float32(1e-8))
        return output,first,second
    raise ValueError(name)


def build(name):
    if name not in NAMES:raise KeyError(name)
    def numpy_prepare(case,x,env):
        state={}
        def run():state["output"]=reference(name,case,x)
        return PreparedCall(run,lambda:state["output"])
    def mojo_prepare(case,x,env):
        from ..implementations.mojo import prepare
        return prepare(name,case,x,env)
    implementations={"numpy":Implementation("numpy",numpy_prepare),"mojo":Implementation("mojo",mojo_prepare)}
    if name=="histogram":
        from .programs import _device
        from ..implementations.v2_program import bind
        def prepare_bins(backend,case,x,env):
            device=_device(env,backend); indices=x["indices"]
            if device!="cpu":
                import torch
                if device.startswith("mlu"):import torch_mlu
                indices=torch.from_numpy(indices).to(device)
            result=bind("histogram_from_indices",backend,(indices,case.parameters["columns"]),env,device)
            if device.startswith("cuda"):
                get=result.result;result.result=lambda:get().cpu().numpy()
            return result
        from functools import partial
        for backend in ("weft","mojo","triton","cutile","bangc"):
            key="intent-v2:"+backend;implementations[key]=Implementation(key,partial(prepare_bins,backend))
    from ..references.aten_rows import registrations
    implementations.update(registrations("cpu/" + name))
    return Workload("cpu/"+name,[Case("small",{"rows":16,"columns":32}),Case("large",{"rows":512,"columns":1024})],
                    lambda case,seed:inputs(name,case,seed),
                    implementations,
                    oracle=lambda case,x:reference(name,case,x),atol=1e-4,rtol=1e-4,family="cpu/"+name)
