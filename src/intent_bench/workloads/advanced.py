"""Small independent cases complement the retained production profiles."""
from ..model import Case,Implementation,PreparedCall,Workload

from ..catalog import ADVANCED_OPERATIONS as NAMES


def data(name,case,seed):
    import torch
    generator=torch.Generator().manual_seed(seed)
    def normal(shape,dtype):return torch.randn(shape,generator=generator,dtype=dtype)*.2
    size=case.parameters['size']
    if name=='mamba3_siso_step':
        b,h,qh,d,dv=size,16,4,32,64
        q=normal((b,qh,d),torch.bfloat16);k=normal(q.shape,q.dtype);v=normal((b,h,dv),torch.bfloat16)
        adt=-torch.rand((b,h),generator=generator)*.1
        dt=torch.rand((b,h),generator=generator)*.1;trap=torch.rand((b,h),generator=generator)*.1
        return (q,k,v,adt,dt,trap,normal((h,d),torch.bfloat16),normal((h,d),torch.bfloat16),
                normal((b,h,d//2),torch.float32),normal((h,),torch.float32),normal(v.shape,torch.bfloat16),
                torch.zeros((b,h,d//2)),torch.zeros((b,h,dv,d)),torch.zeros((b,h,d)),torch.zeros((b,h,dv)))
    if name=='mamba3_siso_forward':
        b,s,h,qh,d,dv=1,size,16,4,32,64
        q=normal((b,s,qh,d),torch.bfloat16);k=normal(q.shape,q.dtype);v=normal((b,s,h,dv),torch.bfloat16)
        adt=-torch.rand((b,h,s),generator=generator)*.1
        dt=torch.rand((b,h,s),generator=generator)*.1;trap=torch.rand((b,h,s),generator=generator)*.1
        return (q,k,v,adt,dt,trap,normal((h,d),torch.bfloat16),normal((h,d),torch.bfloat16),
                normal((b,s,h,d//2),torch.float32),normal((h,),torch.float32),normal(v.shape,torch.bfloat16),
                torch.empty((b,s,h,d),dtype=torch.bfloat16),torch.empty((b,s,h,d),dtype=torch.bfloat16),
                torch.empty((b,h,s)),torch.empty((b,h,s)),torch.empty((b,h,s)))
    if name=='paged_gqa_f16_reduction':
        b,h,kh,d,page=2,4,1,32,16
        count=(size+page-1)//page
        q=normal((b,h,d),torch.float16);k=normal((b*count,page,kh,d),torch.float16);v=normal(k.shape,torch.float16)
        offsets=torch.arange(b+1,dtype=torch.int32)*count
        indices=torch.randperm(b*count,generator=generator).int()
        lengths=torch.full((b,),size,dtype=torch.int32)
        splits=torch.tensor([batch*count+part*count//8 for batch in range(b) for part in range(8)]+[b*count],dtype=torch.int32)
        return q,k,v,offsets,indices,lengths,splits,d**-.5
    if name=='scaled_fp8_gemm':return normal((size,64),torch.float32).to(torch.float8_e4m3fn),normal((64,128),torch.float32).to(torch.float8_e4m3fn),1.,1.
    if name=='fused_linear_cross_entropy':return normal((size,64),torch.bfloat16),normal((128,64),torch.bfloat16),torch.randint(128,(size,),generator=generator)
    if name=='flash_varlen_attention':
        q=normal((size*2,32),torch.float16)
        return q,normal(q.shape,q.dtype),normal(q.shape,q.dtype),torch.tensor([size,size],dtype=torch.int32),torch.tensor([0,size,size*2],dtype=torch.int32),32**-.5
    if name=='chunk_retention':
        shape=(1,size,2,32)
        return *(normal(shape,torch.float16) for _ in range(3)),32**-.5
    shape=(1,2,size,32);q,k,v=(normal(shape,torch.float16) for _ in range(3))
    return (q,k,v,normal(shape[:-1],torch.float32),32**-.5) if name=='flash_attention_bias' else (q,k,v,32**-.5)


def reference(name,args):
    from ..references import advanced
    if name=='mamba3_siso_step':return advanced.mamba3_step(*args)
    if name=='mamba3_siso_forward':
        output=advanced.mamba3_forward(*args)
        return output,tuple(args[-5:])
    if name=='paged_gqa_f16_reduction':
        from ..corpus.cpu.baselines.pytorch.cpu_runtime import _paged_gqa_decode
        return _paged_gqa_decode(*args[:6],args[-1]).half()
    if name=='fused_linear_cross_entropy':return advanced.linear_cross_entropy(*args)
    if name=='scaled_fp8_gemm':return advanced.scaled_fp8_gemm(*args)
    if name=='flash_attention_bias':return advanced.flash_attention_bias(*args)
    if name=='flash_varlen_attention':return advanced.flash_varlen_attention(*args)
    if name=='chunk_retention':return advanced.retention(*args)
    return advanced.flash_attention_causal(*args)


def build(name):
    if name not in NAMES:raise KeyError(name)
    def prepare(case,args,env):
        from ..implementations.advanced_mojo import prepare
        call=prepare(name,args,env)
        if name=='mamba3_siso_forward':
            original=call.result
            call.result=lambda:(original(),tuple(args[-5:]))
        return call
    def torch_prepare(case,args,env):
        import torch
        torch.set_num_threads(env.threads)
        state={}
        def run():state['output']=reference(name,args)
        return PreparedCall(run,lambda:state['output'],scopes=('prepared_call','host_call'))
    def intent_prepare(case,args,env):
        from ..implementations.advanced_intent import prepare
        return prepare(name,args,env)
    def check(actual,expected):
        from ..numeric import compare_outputs
        from ..implementations.context import Tolerance
        compare_outputs(actual,expected,Tolerance(atol=2e-3,rtol=2e-2))
    return Workload('advanced/'+name,[Case('small',{'size':16}),Case('medium',{'size':128})],
                    lambda case,seed:data(name,case,seed),{'mojo':Implementation('mojo',prepare),'torch':Implementation('torch',torch_prepare),
                                                        'intent:mojo':Implementation('intent:mojo',intent_prepare)},
                    oracle=lambda case,args:reference(name,args),check=check,atol=2e-3,rtol=2e-2,family='advanced/'+name)
