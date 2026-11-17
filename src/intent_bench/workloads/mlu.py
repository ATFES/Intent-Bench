import numpy as np
from ..model import Case,Implementation,Workload

NAMES=("relu","softmax","matmul","layer_norm")


def build(name):
    if name not in NAMES:raise KeyError(name)
    def inputs(case,seed):
        rng=np.random.default_rng(seed);p=case.parameters
        a=rng.normal(size=(p["rows"],p["columns"])).astype("float16")
        return (a,rng.normal(size=(p["columns"],p.get('output_columns',p['columns']))).astype("float16")) if name=="matmul" else (a,)
    def oracle(case,arrays):
        a=arrays[0].astype("float32")
        if name=="relu":return np.maximum(a,0).astype("float16")
        if name=="softmax":
            exp=np.exp(a-a.max(-1,keepdims=True));return (exp/exp.sum(-1,keepdims=True)).astype("float16")
        if name=="layer_norm":
            centered=a-a.mean(-1,keepdims=True);return (centered/np.sqrt((centered**2).mean(-1,keepdims=True)+1e-5)).astype("float16")
        return (a@arrays[1].astype("float32")).astype("float16")
    def prepare(case,arrays,env):
        from ..implementations.native_libraries import cnnl
        return cnnl(name,case,arrays,env)
    def generated(case,arrays,env):
        from ..loading import intent_environment
        intent=intent_environment(env)
        import torch
        from ..corpus import PROJECT_ROOT
        from ..implementations.context import Context
        from ..implementations.bangc import prepare_program
        context=Context(env.compiler,PROJECT_ROOT,None,
                        'bangc',environment=env)
        x=torch.from_numpy(arrays[0])
        if name=='relu':
            from intent.examples.kernels.activation.pointwise import relu_forward
            program=context.compile(relu_forward)
            arguments=(x,)
        elif name=='softmax':
            from intent.examples.kernels.normalization.softmax import stable_softmax_f16
            program=context.compile(stable_softmax_f16)
            arguments=(x,)
        elif name=='matmul':
            from intent.examples.kernels.contraction.gemm import gemm,Activation
            program=context.compile(gemm,constexprs={'ACTIVATION':Activation.NONE})
            arguments=(x,torch.from_numpy(arrays[1]))
        else:
            from intent.examples.kernels.normalization.layer_norm import layer_norm_f16
            program=context.compile(layer_norm_f16)
            arguments=(x,torch.ones(x.shape[-1],dtype=torch.float16),
                       torch.zeros(x.shape[-1],dtype=torch.float16),1/x.shape[-1],1e-5)
        return prepare_program(context,program,arguments)
    return Workload("mlu/"+name,[Case("small",{"rows":16,"columns":128}),Case("large",{"rows":256,"columns":1024})],
                    inputs,{"cnnl":Implementation("cnnl",prepare),'intent:bangc':Implementation('intent:bangc',generated)},
                    oracle=oracle,atol=2e-2,rtol=2e-2,family="mlu/"+name)
