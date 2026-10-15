"""The original six RVV cases with independent Weft and Intent preparation."""
import random
import struct
import numpy as np
from ..model import Case,Implementation,Workload


def q4_projection_reference(weights,activation):
    n,g,_=weights.shape
    d=weights[...,:2].copy().view('<f2').reshape(n,g).astype('float32')
    dmin=weights[...,2:4].copy().view('<f2').reshape(n,g).astype('float32')
    packed=weights[...,4:16].astype('int32');qs=weights[...,16:]
    scales=np.empty((n,g,8),dtype='float32');minimum=np.empty_like(scales)
    for j in range(8):
        if j<4:scales[...,j]=packed[...,j]&63;minimum[...,j]=packed[...,j+4]&63
        else:
            scales[...,j]=(packed[...,j+4]&15)|((packed[...,j-4]>>6)<<4)
            minimum[...,j]=(packed[...,j+4]>>4)|((packed[...,j]>>6)<<4)
    dequant=np.empty((n,g,256),dtype='float32')
    for j in range(8):
        values=(qs[...,j//2*32:(j//2+1)*32] >> (4*(j%2))) & 15
        dequant[...,j*32:(j+1)*32]=d[...,None]*scales[...,j,None]*values-dmin[...,None]*minimum[...,j,None]
    blocks=activation.reshape(g,256);indices=np.abs(blocks).argmax(-1);extreme=blocks[np.arange(g),indices]
    inverse=np.divide(np.float32(-127),extreme,out=np.zeros_like(extreme),where=extreme!=0)
    scale=np.divide(np.float32(1),inverse,out=np.zeros_like(inverse),where=inverse!=0)
    quantized=np.clip(np.rint(blocks*inverse[:,None]),-128,127)*scale[:,None]
    return np.einsum('ngk,gk->n',dequant,quantized,dtype='float64').astype('float32')


def build(row):
    name=row['kernel'];projection=name.startswith('q4_k');matrix=name.startswith('i8_')
    linear=name=='causal_linear_attention_f32';ime=matrix or name.endswith('_ime')
    def inputs(case,seed):
        rng=np.random.default_rng(seed)
        if projection:
            rng=random.Random(seed);n=k=4096;g=k//256
            packed=bytearray(rng.randbytes(n*g*144))
            for offset in range(0,len(packed),144):struct.pack_into('<ee',packed,offset,rng.uniform(.001,.004),rng.uniform(.002,.008))
            return np.frombuffer(packed,dtype='uint8').reshape(n,g,144),np.array([rng.uniform(-1,1) for _ in range(k)],dtype='float32').reshape(g,256)
        if matrix:
            m=1 if name=='i8_gemv_bias' else 128
            return rng.integers(-128,128,(m,4096),dtype='int8'),rng.integers(-128,128,(4096,4096),dtype='int8'),rng.integers(-1024,1024,(4096,),dtype='int32')
        data=tuple(rng.normal(size=(8,128,32)).astype('float32') for _ in range(3))
        return data if linear else (*data,32**-.5)
    def source(case,args,env):
        from ..implementations.weft_source import prepare
        prefix='corpus/cpu/baselines/weft/'
        if projection:
            n,g,_=args[0].shape
            return prepare((prefix+'tianchenrv/contraction/q4_k_projection/q4_k_projection.py','production_mul_mat_q4_k'),args,
                [{'shape':[n],'dtype':'float32'}],[{'buffer':0},{'buffer':1},{'workspace':g*292},{'buffer':2}],{'M':1,'N':n,'K':g*256},env,ime=ime,matrix_extension=False)
        if matrix:
            a,b,bias=args;m,k=a.shape;n=b.shape[1]
            return prepare((prefix+'intentdsl/contraction/i8/i8.py','gemm_bias'),args,[{'shape':[m,n],'dtype':'int32'}],
                [{'buffer':i} for i in range(4)],{'M':m,'N':n,'K':k},env,ime=ime,meta=(('MR',1 if m==1 else 4),))
        b,s,d=args[0].shape;dv=args[2].shape[-1]
        family='linear' if linear else 'causal';arguments=args if linear else (*args[:3],np.array([args[3]],dtype='float32'))
        outputs=[{'shape':[b,s,dv],'dtype':'float32'}]
        if linear:outputs.append({'shape':[b,d,dv],'dtype':'float32'})
        bindings=[{'buffer':i} for i in ([0,1,2,3,4] if linear else [0,1,2,4,3])]
        return prepare((prefix+f'intentdsl/attention/{family}/{family}.py','causal_linear_attention' if linear else 'causal_attention'),
            arguments,outputs,bindings,{'B':b,'S':s,'TQ':s,'TK':s,'D':d,'DV':dv},env)
    def generated(case,args,env):
        from ..implementations.weft_intent import prepare
        reference=('kernels.quantization.quantized_projection:quantized_projection' if projection else
                   'kernels.contraction.gemm:gemm_i8' if matrix else 'kernels.streaming.attention_f32:'+name)
        return prepare(reference,args,env,ime=ime)
    def oracle(case,args):
        if projection:return q4_projection_reference(*args)
        # Every i8 product and partial sum fits exactly in a float64 integer.
        if matrix:return (args[0].astype('float64')@args[1].astype('float64')+args[2]).astype('int32')
        q,k,v=args[:3]
        if linear:
            states=np.einsum('bsd,bsv->bsdv',k,v).cumsum(1)
            return np.einsum('bsd,bsdv->bsv',q,states),states[:,-1]
        scores=(q@k.swapaxes(-1,-2))*args[3]
        scores=np.where(np.triu(np.ones((q.shape[1],k.shape[1]),dtype=bool),1),-np.inf,scores)
        probabilities=np.exp(scores-scores.max(-1,keepdims=True))
        return probabilities/probabilities.sum(-1,keepdims=True)@v
    return Workload(row['name'],[Case(row['case'],{'profile':'original'})],inputs,
        {'weft':Implementation('weft',source),'intent:weft':Implementation('intent:weft',generated)},oracle=oracle,
        atol=0 if matrix else 1e-4 if projection else 2e-4,rtol=0 if matrix else 2e-3 if projection else 1e-5,
        family=row['family'],parameter_keys=())
