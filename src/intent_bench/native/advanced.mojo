from std.math import exp, log, sqrt, sin, cos, floor
from std.runtime import initialize_runtime
from max.algorithm.backend.cpu.parallelize import sync_parallelize
from layout import Coord, TileTensor, row_major
from linalg.matmul.cpu import matmul
from std.os import abort

comptime F = Pointer[Float32, MutUntrackedOrigin]
comptime H = Pointer[Float16, MutUntrackedOrigin]
comptime BF = SIMD[DType.bfloat16, 1]
comptime B = Pointer[BF, MutUntrackedOrigin]
comptime U = Pointer[UInt8, MutUntrackedOrigin]
comptime I = Pointer[Int32, MutUntrackedOrigin]
comptime L = Pointer[Int64, MutUntrackedOrigin]


@always_inline
def score(q: H, k: H, qi: Int, ki: Int, d: Int, scale: Float32) -> Float32:
    var total = Float32(0)
    for j in range(d):
        total += Float32(q[unsafe_offset=qi+j])*Float32(k[unsafe_offset=ki+j])
    return total*scale


@export("bench_attention")
def bench_attention(q: H, k: H, v: H, bias: F, output: H, workspace: F,
                    batches: Int64, heads: Int64, queries: Int64, keys: Int64,
                    dimension: Int64, values: Int64, scale: Float32,
                    causal: Int64, biased: Int64, workers: Int64) abi("C"):
    initialize_runtime()
    var bh = Int(batches*heads)
    var nq=Int(queries)
    var nk=Int(keys)
    var d=Int(dimension)
    var dv=Int(values)
    var count=Int(workers)
    def work(w: Int) {imm}:
        for flat in range(w,bh*nq,count):
            var group=flat//nq
            var row=flat%nq
            var extent=min(nk,row+1) if causal else nk
            var maximum=Float32(-1e30)
            for j in range(extent):
                var s=score(q,k,flat*d,(group*nk+j)*d,d,scale)
                if biased:s+=bias[unsafe_offset=group*nk+j]
                workspace[unsafe_offset=w*nk+j]=s
                maximum=max(maximum,s)
            var denominator=Float32(0)
            for j in range(extent):
                var p=exp(workspace[unsafe_offset=w*nk+j]-maximum)
                workspace[unsafe_offset=w*nk+j]=p
                denominator+=p
            for col in range(dv):
                var total=Float32(0)
                for j in range(extent):
                    total+=workspace[unsafe_offset=w*nk+j]*Float32(v[unsafe_offset=(group*nk+j)*dv+col])
                output[unsafe_offset=flat*dv+col]=Float16(total/denominator)
    sync_parallelize(work,count)


@export("bench_varlen_attention")
def bench_varlen_attention(q: H,k: H,v: H,offsets: I,output: H,workspace: F,
                           batches: Int64,total: Int64,dimension: Int64,values: Int64,
                           scale: Float32,workers: Int64) abi("C"):
    initialize_runtime()
    var size=Int(total)
    var d=Int(dimension)
    var dv=Int(values)
    var count=Int(workers)
    def work(w: Int) {imm}:
        for row in range(w,size,count):
            var begin=0
            for batch in range(Int(batches)):
                if row>=Int(offsets[unsafe_offset=batch]) and row<Int(offsets[unsafe_offset=batch+1]):
                    begin=Int(offsets[unsafe_offset=batch])
                    break
            var maximum=Float32(-1e30)
            for j in range(begin,row+1):
                var s=score(q,k,row*d,j*d,d,scale)
                workspace[unsafe_offset=w*size+j]=s
                maximum=max(maximum,s)
            var denominator=Float32(0)
            for j in range(begin,row+1):
                var p=exp(workspace[unsafe_offset=w*size+j]-maximum)
                workspace[unsafe_offset=w*size+j]=p
                denominator+=p
            for col in range(dv):
                var value=Float32(0)
                for j in range(begin,row+1):
                    value+=workspace[unsafe_offset=w*size+j]*Float32(v[unsafe_offset=j*dv+col])
                output[unsafe_offset=row*dv+col]=Float16(value/denominator)
    sync_parallelize(work,count)


@always_inline
def fp8(value: UInt8) -> Float32:
    var bits=Int(value)
    var exponent=(bits>>3)&15
    var mantissa=bits&7
    var significand=Float32(mantissa)/8
    var magnitude=Float32(0)
    if exponent==0:
        magnitude=significand/64
    else:
        magnitude=(1+significand)*Float32(2.0)**(exponent-7)
    return -magnitude if bits&128 else magnitude


@export("bench_fp8_gemm")
def bench_fp8_gemm(a: U,b: U,output: H,left: F,right: F,product: F,
                   rows: Int64,depth: Int64,columns: Int64,
                   left_scale: Float32,right_scale: Float32,workers: Int64) abi("C"):
    initialize_runtime()
    var m=Int(rows)
    var k=Int(depth)
    var n=Int(columns)
    var count=Int(workers)
    def convert(w: Int) {imm}:
        for i in range(w,m*k,count):left[unsafe_offset=i]=fp8(a[unsafe_offset=i])
        for i in range(w,k*n,count):right[unsafe_offset=i]=fp8(b[unsafe_offset=i])
    sync_parallelize(convert,count)
    var lhs=TileTensor(left,row_major(Coord(m,k))).as_immut()
    var rhs=TileTensor(right,row_major(Coord(k,n))).as_immut()
    var result=TileTensor(product,row_major(Coord(m,n)))
    try:matmul[transpose_b=False,b_packed=False](result,lhs,rhs,kernel_type_m=m,num_threads=count)
    except error:abort(String(error))
    def store(w: Int) {imm}:
        for i in range(w,m*n,count):output[unsafe_offset=i]=Float16(product[unsafe_offset=i]*left_scale*right_scale)
    sync_parallelize(store,count)


@export("bench_retention")
def bench_retention(q: H,k: H,v: H,output: H,state: F,
                     batches: Int64,sequence: Int64,heads: Int64,dimension: Int64,values: Int64,
                     scale: Float32,workers: Int64) abi("C"):
    initialize_runtime()
    var s=Int(sequence)
    var h=Int(heads)
    var d=Int(dimension)
    var dv=Int(values)
    var count=Int(workers)
    def work(w: Int) {imm}:
        for bh in range(w,Int(batches)*h,count):
            var batch=bh//h
            var head=bh%h
            var decay=1-exp(Float32(-5-head))
            for i in range(d*dv):state[unsafe_offset=bh*d*dv+i]=0
            for t in range(s):
                var base=(batch*s+t)*h+head
                for i in range(d):
                    for j in range(dv):
                        var offset=(bh*d+i)*dv+j
                        var product=k[unsafe_offset=base*d+i]*v[unsafe_offset=base*dv+j]
                        state[unsafe_offset=offset]=state[unsafe_offset=offset]*decay+Float32(product)
                for j in range(dv):
                    var sum=Float32(0)
                    for i in range(d):
                        var query=Float16(Float32(q[unsafe_offset=base*d+i])*scale)
                        sum+=Float32(query*Float16(state[unsafe_offset=(bh*d+i)*dv+j]))
                    output[unsafe_offset=base*dv+j]=Float16(sum)
    sync_parallelize(work,count)


@export("bench_linear_ce")
def bench_linear_ce(hidden: B,weight: B,target: L,loss: F,mean: F,scratch: B,
                     left: F,right: F,product: F,
                     rows: Int64,depth: Int64,vocabulary: Int64,chunk: Int64,workers: Int64) abi("C"):
    initialize_runtime()
    var m=Int(rows)
    var k=Int(depth)
    var n=Int(vocabulary)
    var count=Int(workers)
    def weights(w: Int) {imm}:
        for i in range(w,n*k,count):right[unsafe_offset=i]=Float32(weight[unsafe_offset=i])
    sync_parallelize(weights,count)
    for begin in range(0,m,Int(chunk)):
        var length=min(Int(chunk),m-begin)
        def load(w: Int) {imm}:
            for i in range(w,length*k,count):left[unsafe_offset=i]=Float32(hidden[unsafe_offset=begin*k+i])
        sync_parallelize(load,count)
        var lhs=TileTensor(left,row_major(Coord(length,k))).as_immut()
        var rhs=TileTensor(right,row_major(Coord(n,k))).as_immut()
        var output=TileTensor(product,row_major(Coord(length,n)))
        try:matmul[transpose_b=True,b_packed=False](output,lhs,rhs,kernel_type_m=length,num_threads=count)
        except error:abort(String(error))
        def normalize(w: Int) {imm}:
            for r in range(w,length,count):
                var maximum=Float32(-1e30)
                for j in range(n):
                    var value=Float32(BF(product[unsafe_offset=r*n+j]))
                    product[unsafe_offset=r*n+j]=value
                    maximum=max(maximum,value)
                var label=Float32(product[unsafe_offset=r*n+Int(target[unsafe_offset=begin+r])])
                var denominator=Float32(0)
                for j in range(n):
                    var value=exp(product[unsafe_offset=r*n+j]-maximum)
                    product[unsafe_offset=r*n+j]=value
                    denominator+=value
                loss[unsafe_offset=begin+r]=maximum+log(denominator)-label
                for j in range(n):scratch[unsafe_offset=r*n+j]=BF(product[unsafe_offset=r*n+j]/denominator)
        sync_parallelize(normalize,count)
    var total=Float32(0)
    for i in range(m):total+=loss[unsafe_offset=i]
    mean[unsafe_offset=0]=total/Float32(m)


@always_inline
def sigmoid(x: Float32) -> Float32:
    return 1/(1+exp(-x))


@export("bench_mamba_step")
def bench_mamba_step(q: B,k: B,v: B,adt: F,dt: F,trap: F,qbias: B,kbias: B,
                     angles: F,residual: F,gate: B,angle_state: F,ssm_state: F,
                     key_state: F,value_state: F,output: B,out_angle: F,out_state: F,
                     out_key: F,workspace: F,batches: Int64,heads: Int64,qheads: Int64,
                     dimension: Int64,values: Int64,workers: Int64) abi("C"):
    initialize_runtime()
    var h=Int(heads)
    var qh=Int(qheads)
    var d=Int(dimension)
    var dv=Int(values)
    var count=Int(workers)
    def work(w: Int) {imm}:
        for bh in range(w,Int(batches)*h,count):
            var head=bh%h
            var qbase=((bh//h)*qh+head//(h//qh))*d
            var current_dt=dt[unsafe_offset=bh]
            for pair in range(d//2):
                var index=bh*(d//2)+pair
                var delta=(2*sigmoid(2*angles[unsafe_offset=index])-1)*Float32(3.141592653589793)*current_dt
                var angle=angle_state[unsafe_offset=index]+delta
                angle-=Float32(6.283185307179586)*floor(angle/Float32(6.283185307179586))
                out_angle[unsafe_offset=index]=angle
                var cosine=cos(angle)
                var sine=sin(angle)
                var q0=Float32(q[unsafe_offset=qbase+2*pair])+Float32(qbias[unsafe_offset=head*d+2*pair])
                var q1=Float32(q[unsafe_offset=qbase+2*pair+1])+Float32(qbias[unsafe_offset=head*d+2*pair+1])
                var k0=Float32(k[unsafe_offset=qbase+2*pair])+Float32(kbias[unsafe_offset=head*d+2*pair])
                var k1=Float32(k[unsafe_offset=qbase+2*pair+1])+Float32(kbias[unsafe_offset=head*d+2*pair+1])
                workspace[unsafe_offset=w*d+2*pair]=Float32(BF(q0*cosine-q1*sine))
                workspace[unsafe_offset=w*d+2*pair+1]=Float32(BF(q0*sine+q1*cosine))
                out_key[unsafe_offset=bh*d+2*pair]=Float32(BF(k0*cosine-k1*sine))
                out_key[unsafe_offset=bh*d+2*pair+1]=Float32(BF(k0*sine+k1*cosine))
            var alpha=exp(adt[unsafe_offset=bh])
            var mix=sigmoid(trap[unsafe_offset=bh])
            var beta=alpha*current_dt*(1-mix)
            var gamma=current_dt*mix
            for i in range(dv):
                var current=Float32(v[unsafe_offset=bh*dv+i])
                var previous=value_state[unsafe_offset=bh*dv+i]
                var projected=Float32(0)
                for j in range(d):
                    var pos=(bh*dv+i)*d+j
                    var state=ssm_state[unsafe_offset=pos]*alpha+(beta*previous)*key_state[unsafe_offset=bh*d+j]+(gamma*current)*out_key[unsafe_offset=bh*d+j]
                    out_state[unsafe_offset=pos]=state
                    projected+=Float32(BF(state))*workspace[unsafe_offset=w*d+j]
                var gating=Float32(gate[unsafe_offset=bh*dv+i])
                output[unsafe_offset=bh*dv+i]=BF((projected+residual[unsafe_offset=head]*current)*gating*sigmoid(gating))
    sync_parallelize(work,count)


@export("bench_mamba_forward")
def bench_mamba_forward(q: B,k: B,v: B,adt: F,dt: F,trap: F,qbias: B,kbias: B,
                        angles: F,residual: F,gate: B,query_store: B,key_store: B,
                        qk_store: F,scale_store: F,gamma_store: F,output: B,
                        state: F,scores: F,decays: F,batches: Int64,sequence: Int64,
                        heads: Int64,qheads: Int64,dimension: Int64,values: Int64,workers: Int64) abi("C"):
    initialize_runtime()
    var s=Int(sequence)
    var h=Int(heads)
    var qh=Int(qheads)
    var d=Int(dimension)
    var dv=Int(values)
    var count=Int(workers)
    def work(w: Int) {imm}:
        for bh in range(w,Int(batches)*h,count):
            var batch=bh//h
            var head=bh%h
            for t in range(s):
                var base=(batch*s+t)*h+head
                var qbase=((batch*s+t)*qh+head//(h//qh))*d
                var gamma=dt[unsafe_offset=bh*s+t]*sigmoid(trap[unsafe_offset=bh*s+t])
                var shift=Float32(0)
                if t+1<s:shift=dt[unsafe_offset=bh*s+t+1]*(1-sigmoid(trap[unsafe_offset=bh*s+t+1]))
                var transition=gamma+shift
                var dot=Float32(0)
                for pair in range(d//2):
                    var angle=angles[unsafe_offset=base*(d//2)+pair]
                    var cosine=cos(angle)
                    var sine=sin(angle)
                    var q0=Float32(q[unsafe_offset=qbase+2*pair])+Float32(qbias[unsafe_offset=head*d+2*pair])
                    var q1=Float32(q[unsafe_offset=qbase+2*pair+1])+Float32(qbias[unsafe_offset=head*d+2*pair+1])
                    var k0=Float32(k[unsafe_offset=qbase+2*pair])+Float32(kbias[unsafe_offset=head*d+2*pair])
                    var k1=Float32(k[unsafe_offset=qbase+2*pair+1])+Float32(kbias[unsafe_offset=head*d+2*pair+1])
                    query_store[unsafe_offset=base*d+2*pair]=BF(q0*cosine-q1*sine)
                    query_store[unsafe_offset=base*d+2*pair+1]=BF(q0*sine+q1*cosine)
                    key_store[unsafe_offset=base*d+2*pair]=BF(k0*cosine-k1*sine)*BF(transition)
                    key_store[unsafe_offset=base*d+2*pair+1]=BF(k0*sine+k1*cosine)*BF(transition)
                    dot+=Float32(BF(q0*k0+q1*k1))
                qk_store[unsafe_offset=bh*s+t]=dot*gamma
                scale_store[unsafe_offset=bh*s+t]=transition
                gamma_store[unsafe_offset=bh*s+t]=gamma
            for i in range(dv*d):state[unsafe_offset=w*dv*d+i]=0
            for begin in range(0,s,64):
                var length=min(64,s-begin)
                var total=Float32(0)
                for r in range(length):
                    total+=adt[unsafe_offset=bh*s+begin+r]
                    decays[unsafe_offset=w*64+r]=total
                for r in range(length):
                    var qr=((batch*s+begin+r)*h+head)*d
                    for t in range(r):
                        var kr=((batch*s+begin+t)*h+head)*d
                        var dot=Float32(0)
                        for j in range(d):dot+=Float32(query_store[unsafe_offset=qr+j])*Float32(key_store[unsafe_offset=kr+j])
                        var decay=min(decays[unsafe_offset=w*64+r]-decays[unsafe_offset=w*64+t],Float32(0))
                        scores[unsafe_offset=w*4096+r*64+t]=Float32(BF(dot*exp(decay)))
                    for i in range(dv):
                        var carried=Float32(0)
                        for j in range(d):carried+=Float32(query_store[unsafe_offset=qr+j])*Float32(BF(state[unsafe_offset=(w*dv+i)*d+j]))
                        carried*=exp(decays[unsafe_offset=w*64+r])
                        var current=Float32(0)
                        for t in range(r):
                            var vi=((batch*s+begin+t)*h+head)*dv+i
                            current+=scores[unsafe_offset=w*4096+r*64+t]*Float32(v[unsafe_offset=vi])
                        var index=((batch*s+begin+r)*h+head)*dv+i
                        var skip=(residual[unsafe_offset=head]+qk_store[unsafe_offset=bh*s+begin+r])*Float32(v[unsafe_offset=index])
                        var gating=Float32(gate[unsafe_offset=index])
                        output[unsafe_offset=index]=BF((carried+current+skip)*gating*sigmoid(gating))
                for i in range(dv):
                    for j in range(d):
                        var update=Float32(0)
                        for t in range(length):
                            var base=(batch*s+begin+t)*h+head
                            var weighted=BF(Float32(v[unsafe_offset=base*dv+i])*exp(total-decays[unsafe_offset=w*64+t]))
                            update+=Float32(weighted)*Float32(key_store[unsafe_offset=base*d+j])
                        var pos=(w*dv+i)*d+j
                        state[unsafe_offset=pos]=state[unsafe_offset=pos]*exp(total)+update
    sync_parallelize(work,count)


@export("bench_paged_attention")
def bench_paged_attention(q: H,k: H,v: H,offsets: I,indices: I,lengths: I,
                          output: H,workspace: F,batches: Int64,query_heads: Int64,
                          key_heads: Int64,page_size: Int64,dimension: Int64,values: Int64,
                          max_length: Int64,scale: Float32,workers: Int64) abi("C"):
    initialize_runtime()
    var h=Int(query_heads)
    var kh=Int(key_heads)
    var page=Int(page_size)
    var d=Int(dimension)
    var dv=Int(values)
    var capacity=Int(max_length)
    var count=Int(workers)
    def work(w: Int) {imm}:
        for bh in range(w,Int(batches)*h,count):
            var batch=bh//h
            var head=(bh%h)//(h//kh)
            var size=Int(lengths[unsafe_offset=batch])
            var maximum=Float32(-1e30)
            for j in range(size):
                var physical=Int(indices[unsafe_offset=Int(offsets[unsafe_offset=batch])+j//page])
                var keyrow=(physical*page+j%page)*kh+head
                var value=score(q,k,bh*d,keyrow*d,d,scale)
                workspace[unsafe_offset=w*capacity+j]=value
                maximum=max(maximum,value)
            var denominator=Float32(0)
            for j in range(size):
                var value=exp(workspace[unsafe_offset=w*capacity+j]-maximum)
                workspace[unsafe_offset=w*capacity+j]=value
                denominator+=value
            for col in range(dv):
                var total=Float32(0)
                for j in range(size):
                    var physical=Int(indices[unsafe_offset=Int(offsets[unsafe_offset=batch])+j//page])
                    var row=(physical*page+j%page)*kh+head
                    total+=workspace[unsafe_offset=w*capacity+j]*Float32(v[unsafe_offset=row*dv+col])
                output[unsafe_offset=bh*dv+col]=Float16(total/denominator)
    sync_parallelize(work,count)
