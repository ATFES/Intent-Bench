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
        from ..implementations.bangc_sequence import LocalSequence
        from ..implementations.bangc import prepare_sequence
        context=Context(env.compiler,PROJECT_ROOT,None,
                        'bangc',environment=env)
        sequence=LocalSequence(context)
        x=torch.from_numpy(arrays[0])
        if name=='relu':
            from intent.examples.kernels.activation.pointwise import relu_forward
            outputs=sequence.add(relu_forward,{'x':x})['output']
        elif name=='softmax':
            from intent.examples.kernels.normalization.softmax import stable_softmax_f16
            outputs=sequence.add(stable_softmax_f16,{'x':x})['y']
        elif name=='matmul':
            from intent.examples.kernels.contraction.gemm import gemm,Activation
            outputs=sequence.add(gemm,{'a':x,'b':torch.from_numpy(arrays[1])},constexprs={'ACTIVATION':Activation.NONE})['c']
        else:
            from intent.examples.kernels.normalization.layer_norm import layer_norm_f16
            outputs=sequence.add(layer_norm_f16,{'x':x,'weight':torch.ones(x.shape[-1],dtype=torch.float16),
                'bias':torch.zeros(x.shape[-1],dtype=torch.float16),'inverse_features':1/x.shape[-1],'epsilon':1e-5})['y']
        return prepare_sequence(sequence,outputs)
    return Workload("mlu/"+name,[Case("small",{"rows":16,"columns":128}),Case("large",{"rows":256,"columns":1024})],
                    inputs,{"cnnl":Implementation("cnnl",prepare),'intent:bangc':Implementation('intent:bangc',generated)},
                    oracle=oracle,atol=2e-2,rtol=2e-2,family="mlu/"+name)
