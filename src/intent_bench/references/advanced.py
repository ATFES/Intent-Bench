"""CPU references for previously unchecked workloads, including declared casts."""
import math
import torch


def attention(q,k,v,scale,*,bias=None,causal=False):
    # Query blocking avoids materializing the full production Q x K score matrix.
    pieces=[]
    for start in range(0,q.shape[-2],64):
        scores=(q[...,start:start+64,:].float()@k.float().transpose(-1,-2))*scale
        if bias is not None:scores=scores+bias.float().unsqueeze(-2)
        if causal:
            query=torch.arange(start,start+scores.shape[-2],device=q.device)
            key=torch.arange(k.shape[-2],device=q.device)
            scores=scores.masked_fill(key[None,:]>query[:,None],-torch.inf)
        probabilities=torch.softmax(scores,-1)
        pieces.append((probabilities@v.float()).to(q.dtype))
    return torch.cat(pieces,-2)


def flash_attention_bias(q,k,v,bias,scale):return attention(q,k,v,scale,bias=bias)


def flash_attention_causal(q,k,v,scale):return attention(q,k,v,scale,causal=True)


def flash_varlen_attention(q,k,v,lengths,offsets,scale):
    output=torch.empty_like(v)
    for begin,end in zip(offsets[:-1],offsets[1:]):
        a,b=int(begin),int(end)
        output[a:b]=attention(q[a:b],k[a:b],v[a:b],scale,causal=True)
    return output


def scaled_fp8_gemm(a,b,left_scale,right_scale):
    return ((a.float()@b.float())*left_scale*right_scale).half()


def linear_cross_entropy(hidden,weight,target,chunk=1024):
    losses=[];scratch=None
    for begin in range(0,hidden.shape[0],chunk):
        logits=(hidden[begin:begin+chunk].float()@weight.float().T).bfloat16().float()
        labels=target[begin:begin+chunk]
        losses.append(torch.logsumexp(logits,-1)-logits[torch.arange(logits.shape[0]),labels])
        scratch=torch.softmax(logits,-1).bfloat16()
    loss=torch.cat(losses)
    return loss,loss.mean().reshape(1),scratch


def retention(q,k,v,scale):
    b,s,h,d=q.shape;dv=v.shape[-1]
    state=torch.zeros((b,h,d,dv),dtype=torch.float32)
    decay=1-torch.exp(-5-torch.arange(h,dtype=torch.float32))
    output=torch.empty_like(v)
    for i in range(s):
        # Source product is f16, carry is f32, readback of the carry is f16.
        state=state*decay[None,:,None,None]+(k[:,i,:,:,None]*v[:,i,:,None,:]).float()
        query=(q[:,i].float()*scale).half()
        output[:,i]=(query[:,:,:,None]*state.half()).sum(-2,dtype=torch.float16)
    return output


def _rotate(value,angle):
    pairs=value.reshape(*value.shape[:-1],-1,2)
    c,s=torch.cos(angle),torch.sin(angle)
    return torch.stack((pairs[...,0]*c-pairs[...,1]*s,pairs[...,0]*s+pairs[...,1]*c),-1).flatten(-2).bfloat16()


def mamba3_step(q,k,v,adt,dt,trap,qbias,kbias,angles,residual,gate,angle_state,ssm_state,key_state,value_state):
    heads=v.shape[-2];group=heads//q.shape[-2]
    query=q.float().repeat_interleave(group,dim=-2)+qbias.float()
    key=k.float().repeat_interleave(group,dim=-2)+kbias.float()
    angle=(2*torch.sigmoid(2*angles)-1)*math.pi*dt[...,None]+angle_state
    angle=angle-2*math.pi*torch.floor(angle/(2*math.pi))
    rq,rk=_rotate(query,angle),_rotate(key,angle)
    alpha=torch.exp(adt);mix=torch.sigmoid(trap)
    beta=alpha*dt*(1-mix);gamma=dt*mix
    state=(ssm_state*alpha[...,None,None]+(beta[...,None]*value_state)[...,None]*key_state[...,None,:]
           +(gamma[...,None]*v.float())[...,None]*rk.float()[...,None,:])
    projected=(state.bfloat16().float()@rq.float().unsqueeze(-1)).squeeze(-1)
    gated=(projected+residual[None,:,None]*v.float())*gate.float()*torch.sigmoid(gate.float())
    return gated.bfloat16(),angle,state,rk.float()


def mamba3_forward(q,k,v,adt,dt,trap,qbias,kbias,angles,residual,gate,query_store,key_store,qk_store,scale_store,gamma_store):
    b,s,h,dv=v.shape;d=q.shape[-1];group=h//q.shape[-2]
    query=q.float().repeat_interleave(group,dim=-2)+qbias.float()
    key=k.float().repeat_interleave(group,dim=-2)+kbias.float()
    gamma=dt*torch.sigmoid(trap)
    shifted=torch.cat((dt[:,:,1:]*(1-torch.sigmoid(trap[:,:,1:])),torch.zeros_like(dt[:,:,:1])),dim=-1)
    transition=gamma+shifted
    query_store.copy_(_rotate(query,angles))
    key_store.copy_(_rotate(key,angles)*transition.permute(0,2,1).unsqueeze(-1).bfloat16())
    qp=query.reshape(b,s,h,d//2,2);kp=key.reshape_as(qp)
    qk=((qp[...,0]*kp[...,0]+qp[...,1]*kp[...,1]).bfloat16().float().sum(-1).permute(0,2,1)*gamma)
    qk_store.copy_(qk);scale_store.copy_(transition);gamma_store.copy_(gamma)
    output=torch.empty_like(v);state=torch.zeros((b,h,dv,d),dtype=torch.float32)
    for start in range(0,s,64):
        end=min(start+64,s);width=end-start
        qb=query_store[:,start:end].permute(0,2,1,3).float()
        kb=key_store[:,start:end].permute(0,2,1,3).float()
        vb=v[:,start:end].permute(0,2,1,3).float()
        decay=adt[:,:,start:end].cumsum(-1);total=adt[:,:,start:end].sum(-1)
        carried=(qb@state.bfloat16().float().transpose(-1,-2))*torch.exp(decay)[...,None]
        scores=(qb@kb.transpose(-1,-2))*torch.exp(torch.clamp(decay[...,None]-decay[...,None,:],max=0))
        lower=torch.arange(width)[:,None]>torch.arange(width)[None,:]
        scores=scores.masked_fill(~lower,0).bfloat16().float()
        current=scores@vb
        skip=(residual[None,:,None]+qk[:,:,start:end])[...,None]*vb
        g=gate[:,start:end].permute(0,2,1,3).float()
        output[:,start:end]=((carried+current+skip)*g*torch.sigmoid(g)).bfloat16().permute(0,2,1,3)
        weighted=(vb*torch.exp(total[...,None]-decay)[...,None]).bfloat16().float()
        state=state*torch.exp(total)[...,None,None]+weighted.transpose(-1,-2)@kb
    return output
