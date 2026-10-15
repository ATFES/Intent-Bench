"""Native Mojo expressions for advanced workloads; input casts are part of the call."""
from pathlib import Path
from types import SimpleNamespace
import torch

from ..model import PreparedCall, ImplementationMissing
from ..corpus.cpu.baselines.mojo.source import prepare_source,view,scalar


def bind(entry,tensors,scalars,outputs,env):
    declarations=[];arguments=[];dimension=1
    types={torch.float32:'f32',torch.float16:'f16',torch.bfloat16:'bf16',torch.int32:'i32',torch.int64:'i64',torch.uint8:'u8'}
    for i,tensor in enumerate(tensors):
        dims=tuple(range(dimension,dimension+tensor.ndim));dimension+=tensor.ndim
        declarations.append(view('buffer'+str(i),dims,dtype=types[tensor.dtype],compact=True))
        arguments.append(tensor)
    for i,(dtype,value) in enumerate(scalars):
        declarations.append(scalar('scalar'+str(i),dtype));arguments.append(value)
    target=SimpleNamespace(workers=env.threads,executable=env.options.get('mojo'))
    call=prepare_source(Path(__file__).parents[1]/'native/advanced.mojo',entry,declarations,target,tuple(arguments),required_workers=None)
    return PreparedCall(call.launch,lambda:outputs[0] if len(outputs)==1 else tuple(outputs),
                        native_measure=lambda settings:call.samples(settings.samples,settings.warmup),
                        timing='cpu_native_monotonic',metadata={'native_language':'Mojo','workers':env.threads})


def prepare(name,args,env):
    workers=env.threads
    if name in {'flash_attention_bias','flash_attention_inline','flash_attention_select','flash_attention_full_causal_stream'}:
        q,k,v=args[:3];b,h,nq,d=q.shape;nk=k.shape[-2];dv=v.shape[-1]
        biased=name=='flash_attention_bias'
        bias=args[3] if biased else torch.zeros((b,h,nk),dtype=torch.float32)
        output=torch.empty((b,h,nq,dv),dtype=torch.float16)
        scratch=torch.empty((workers,nk),dtype=torch.float32)
        scalars=[('i64',x) for x in (b,h,nq,nk,d,dv)]+[('f32',args[-1])]+[('i64',x) for x in (int(not biased),int(biased),workers)]
        return bind('bench_attention',[q,k,v,bias,output,scratch],scalars,[output],env)
    if name=='flash_varlen_attention':
        q,k,v,lengths,offsets,scale=args;size,d=q.shape;dv=v.shape[-1]
        output=torch.empty_like(v);scratch=torch.empty((workers,size),dtype=torch.float32)
        scalars=[('i64',x) for x in (len(lengths),size,d,dv)]+[('f32',scale),('i64',workers)]
        return bind('bench_varlen_attention',[q,k,v,offsets,output,scratch],scalars,[output],env)
    if name=='scaled_fp8_gemm':
        a,b,scale_a,scale_b=args;m,k=a.shape;n=b.shape[1]
        output=torch.empty((m,n),dtype=torch.float16)
        buffers=[torch.empty(shape,dtype=torch.float32) for shape in ((m,k),(k,n),(m,n))]
        scalars=[('i64',x) for x in (m,k,n)]+[('f32',scale_a),('f32',scale_b),('i64',workers)]
        return bind('bench_fp8_gemm',[a.view(torch.uint8),b.view(torch.uint8),output,*buffers],scalars,[output],env)
    if name=='chunk_retention':
        q,k,v,scale=args;b,s,h,d=q.shape;dv=v.shape[-1]
        output=torch.empty_like(v);state=torch.empty((b,h,d,dv),dtype=torch.float32)
        scalars=[('i64',x) for x in (b,s,h,d,dv)]+[('f32',scale),('i64',workers)]
        return bind('bench_retention',[q,k,v,output,state],scalars,[output],env)
    if name=='fused_linear_cross_entropy':
        hidden,weight,target=args[:3];m,k=hidden.shape;n=weight.shape[0]
        chunk=min(1024,m)
        loss=torch.empty(m,dtype=torch.float32);mean=torch.empty(1,dtype=torch.float32)
        scratch=torch.empty((chunk,n),dtype=torch.bfloat16)
        buffers=[torch.empty(shape,dtype=torch.float32) for shape in ((chunk,k),(n,k),(chunk,n))]
        return bind('bench_linear_ce',[hidden,weight,target,loss,mean,scratch,*buffers],
                    [('i64',x) for x in (m,k,n,chunk,workers)],[loss,mean,scratch],env)
    if name=='mamba3_siso_step':
        q,k,v=args[:3];b,h,dv=v.shape;qh=q.shape[1];d=q.shape[-1]
        output=torch.empty_like(v)
        angle=torch.empty((b,h,d//2),dtype=torch.float32)
        state=torch.empty((b,h,dv,d),dtype=torch.float32)
        key=torch.empty((b,h,d),dtype=torch.float32)
        scratch=torch.empty((workers,d),dtype=torch.float32)
        return bind('bench_mamba_step',list(args)+[output,angle,state,key,scratch],
                    [('i64',x) for x in (b,h,qh,d,dv,workers)],[output,angle,state,key],env)
    if name=='mamba3_siso_forward':
        q,k,v=args[:3];b,s,h,dv=v.shape;qh=q.shape[2];d=q.shape[-1]
        output=torch.empty_like(v)
        state=torch.empty((workers,dv,d),dtype=torch.float32)
        scores=torch.empty((workers,64,64),dtype=torch.float32)
        decays=torch.empty((workers,64),dtype=torch.float32)
        return bind('bench_mamba_forward',list(args)+[output,state,scores,decays],
                    [('i64',x) for x in (b,s,h,qh,d,dv,workers)],[output],env)
    if name=='paged_gqa_f16_reduction':
        q,k,v,offsets,indices,lengths,splits,scale=args
        b,h,d=q.shape;_,page,kh,_=k.shape;dv=v.shape[-1]
        maximum=int(lengths.max())
        output=torch.empty((b,h,dv),dtype=torch.float16)
        workspace=torch.empty((workers,maximum),dtype=torch.float32)
        return bind('bench_paged_attention',[q,k,v,offsets,indices,lengths,output,workspace],
                    [('i64',x) for x in (b,h,kh,page,d,dv,maximum)]+[('f32',scale),('i64',workers)],[output],env)
    raise ImplementationMissing(f'native Mojo binding for {name} is not implemented')
