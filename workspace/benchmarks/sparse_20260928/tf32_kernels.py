"""Explicit Tensor-Core candidates; FP32 storage and accumulation throughout.

The IEEE production kernels are unchanged. TF32 operands are explicitly rounded
to nearest on SM120. Frozen weights and short Toeplitz filter blocks are prepared
outside steady-state timing. All calls consume current inputs.
"""
import torch
import triton
import triton.language as tl
from quadratic_tiled import upper_weights, _finish, _finish_columns
from triton_scan import affine_combine


@triton.jit
def round_tf32(v):
    return tl.inline_asm_elementwise('cvt.rn.tf32.f32 $0, $1;', '=r,r', [v],
                                    dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def tc_contract(Y,W,P,O,M:tl.constexpr,A:tl.constexpr,NT:tl.constexpr,BIAS:tl.constexpr,
                BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GROUP:tl.constexpr,
                TRI:tl.constexpr,PREC:tl.constexpr,ROUND:tl.constexpr):
    pid=tl.program_id(0);nm=tl.cdiv(M,BM)
    group=pid//(GROUP*NT);first=group*GROUP;gm=tl.minimum(nm-first,GROUP)
    pm=first+(pid%(GROUP*NT))%gm;pn=(pid%(GROUP*NT))//gm
    r=pm*BM+tl.arange(0,BM);n=pn*BN+tl.arange(0,BN);k=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    end=A
    if TRI:end=tl.minimum(A,(pn+1)*BN)
    for base in range(0,end,BK):
        a=tl.load(Y+r[:,None]*A+(base+k[None,:]),(r[:,None]<M)&(base+k[None,:]<A),0)
        w=tl.load(W+(base+k[:,None])*A+n[None,:],(base+k[:,None]<A)&(n[None,:]<A),0)
        if ROUND:a=round_tf32(a);w=round_tf32(w)
        acc=tl.dot(a,w,acc,input_precision=PREC)
    yn=tl.load(Y+r[:,None]*A+n[None,:],(r[:,None]<M)&(n[None,:]<A),0)
    v=tl.sum(acc*yn+yn,1)
    if NT==1:tl.store(O+r,v+BIAS,r<M)
    else:tl.store(P+r*NT+pn,v,r<M)


@triton.jit
def tc_pair(Y,W,I,J,P,M:tl.constexpr,A:tl.constexpr,PAIRS:tl.constexpr,
            BM:tl.constexpr,S:tl.constexpr,PREC:tl.constexpr,ROUND:tl.constexpr):
    pm=tl.program_id(0);pair=tl.program_id(1)
    ti=tl.load(I+pair);tj=tl.load(J+pair)
    r=pm*BM+tl.arange(0,BM);k=ti*S+tl.arange(0,S);n=tj*S+tl.arange(0,S)
    a=tl.load(Y+r[:,None]*A+k[None,:],(r[:,None]<M)&(k[None,:]<A),0)
    w=tl.load(W+k[:,None]*A+n[None,:],(k[:,None]<A)&(n[None,:]<A),0)
    if ROUND:a=round_tf32(a);w=round_tf32(w)
    z=tl.dot(a,w,input_precision=PREC)
    yn=tl.load(Y+r[:,None]*A+n[None,:],(r[:,None]<M)&(n[None,:]<A),0)
    v=tl.sum(z*yn,1)
    if ti==tj:v+=tl.sum(yn,1)
    tl.store(P+pair*M+r,v,r<M)


TC_TILES=[(16,32,32,4,2),(32,32,32,4,2),(32,64,32,4,2),(64,64,32,4,2),
          (32,128,32,4,3),(64,128,32,4,3),(128,64,32,4,3),(128,128,32,8,3),
          (32,64,64,4,3),(64,64,64,4,3),(64,128,64,4,3),(128,128,64,8,3),
          (64,256,32,8,3),(128,64,64,4,3),(32,128,64,4,4),(64,64,32,8,2),
          (16,64,32,4,2),(32,64,32,4,4),(64,128,32,4,4),(64,128,64,8,3)]
TC_PAIR_TILES=[(16,32,4),(32,32,4),(64,32,4),(32,64,4),(64,64,4),(32,128,4),(64,128,8)]


class TensorDrive:
    def __init__(self,y,w,bias,backend='tile',config=3,triangle=True,precision='tf32',rounding=True,group=8):
        assert y.dtype==w.dtype==torch.float32 and y.is_cuda and y.is_contiguous()
        self.y=y;self.M,self.A=y.shape;self.bias=float(bias)
        self.backend=backend;self.config=config;self.triangle=triangle
        self.precision=precision;self.rounding=rounding;self.group=group
        self.w=upper_weights(w) if triangle or backend=='pair' else w
        self.out=torch.empty(self.M,device=y.device,dtype=y.dtype)
        if backend=='tile':
            self.tile=TC_TILES[config];self.nt=triton.cdiv(self.A,self.tile[1])
            self.partial=torch.empty((self.M,self.nt),device=y.device,dtype=y.dtype)
        else:
            self.tile=TC_PAIR_TILES[config];nt=triton.cdiv(self.A,self.tile[1])
            ij=[(i,j) for j in range(nt) for i in range(j+1)]
            self.nt=len(ij)
            self.ii=torch.tensor([i for i,j in ij],device=y.device,dtype=torch.int32)
            self.jj=torch.tensor([j for i,j in ij],device=y.device,dtype=torch.int32)
            self.partial=torch.empty((self.nt,self.M),device=y.device,dtype=y.dtype)

    def __call__(self,y=None):
        if y is None:y=self.y
        if self.backend=='tile':
            bm,bn,bk,nw,ns=self.tile
            self.compiled=tc_contract[(triton.cdiv(self.M,bm)*self.nt,)](y,self.w,self.partial,self.out,
                self.M,self.A,self.nt,self.bias,bm,bn,bk,self.group,self.triangle,self.precision,self.rounding,
                num_warps=nw,num_stages=ns)
            if self.nt>1:_finish[(triton.cdiv(self.M,128),)](self.partial,self.out,self.M,self.nt,
                self.bias,triton.next_power_of_2(self.nt),128,num_warps=4)
        else:
            bm,s,nw=self.tile
            self.compiled=tc_pair[(triton.cdiv(self.M,bm),self.nt)](y,self.w,self.ii,self.jj,self.partial,
                self.M,self.A,self.nt,bm,s,self.precision,self.rounding,num_warps=nw,num_stages=2)
            _finish_columns[(triton.cdiv(self.M,32),)](self.partial,self.out,self.M,self.nt,self.bias,
                triton.next_power_of_2(self.nt),32,num_warps=4 if self.nt<=256 else 8)
        return self.out


@triton.jit
def filter_local(X,H,Z,EA,EC,DA,DC,K:tl.constexpr,T:tl.constexpr,C:tl.constexpr,
                 L:tl.constexpr,ROWS:tl.constexpr,PREC:tl.constexpr,ROUND:tl.constexpr):
    nk=tl.program_id(0);base=tl.program_id(1)*ROWS
    channel=nk%K
    rows=base+tl.arange(0,ROWS);t=tl.arange(0,L)
    x=tl.load(X+nk*T+rows[:,None]*L+t[None,:],(rows[:,None]<C)&(rows[:,None]*L+t[None,:]<T),0)
    h=tl.load(H+channel*L*L+t[:,None]*L+t[None,:])
    a=tl.load(DA+channel);c=tl.load(DC+channel)
    pa=tl.exp(tl.log(a)*(L-1-t));pc=tl.exp(tl.log(c)*(L-1-t))
    enda=tl.sum(x*pa[None,:],1);endc=tl.sum(x*pc[None,:],1)
    tl.store(EA+nk*C+rows,enda,rows<C);tl.store(EC+nk*C+rows,endc,rows<C)
    if ROUND:x=round_tf32(x);h=round_tf32(h)
    local=tl.dot(x,h,input_precision=PREC)
    tl.store(Z+nk*C*L+rows[:,None]*L+t[None,:],local,rows[:,None]<C)


@triton.jit
def filter_boundaries(EA,EC,SA,SC,DA,DC,K:tl.constexpr,C:tl.constexpr,L:tl.constexpr,NC:tl.constexpr):
    nk=tl.program_id(0);i=tl.arange(0,NC);channel=nk%K
    aa=tl.exp(tl.log(tl.load(DA+channel))*L);cc=tl.exp(tl.log(tl.load(DC+channel))*L)
    ea=tl.load(EA+nk*C+i,i<C,0);ec=tl.load(EC+nk*C+i,i<C,0)
    _,sa=tl.associative_scan((tl.full((NC,),1.,tl.float32)*aa,ea),0,affine_combine)
    _,sc=tl.associative_scan((tl.full((NC,),1.,tl.float32)*cc,ec),0,affine_combine)
    tl.store(SA+nk*C+i,sa,i<C);tl.store(SC+nk*C+i,sc,i<C)


@triton.jit
def filter_join(Z,SA,SC,Y,DA,DC,OM,K:tl.constexpr,T:tl.constexpr,C:tl.constexpr,L:tl.constexpr,
                BT:tl.constexpr,BK:tl.constexpr):
    t=tl.program_id(0)*BT+tl.arange(0,BT)
    k=tl.program_id(1)*BK+tl.arange(0,BK);b=tl.program_id(2)
    chunk=t//L;offset=t%L
    a=tl.load(DA+k,k<K,1);c=tl.load(DC+k,k<K,1);om=tl.load(OM+k,k<K,0)
    sa=tl.load(SA+(b*K+k[None,:])*C+chunk[:,None]-1,(k[None,:]<K)&(t[:,None]<T)&(chunk[:,None]>0),0)
    sc=tl.load(SC+(b*K+k[None,:])*C+chunk[:,None]-1,(k[None,:]<K)&(t[:,None]<T)&(chunk[:,None]>0),0)
    local=tl.load(Z+(b*K+k[None,:])*C*L+t[:,None],(k[None,:]<K)&(t[:,None]<T),0)
    pa=tl.exp(tl.log(a[None,:])*(offset[:,None]+1));pc=tl.exp(tl.log(c[None,:])*(offset[:,None]+1))
    v=local+om[None,:]*(sa*pa-sc*pc)
    tl.store(Y+(b*T+t[:,None])*K+k[None,:],v,(t[:,None]<T)&(k[None,:]<K))


class TensorFilter:
    def __init__(self,p,block=32,rows=32,precision='tf32',rounding=True):
        self.p=p;self.L=block;self.rows=rows;self.precision=precision;self.rounding=rounding
        self.B,self.K,self.T=p.x.shape;self.C=triton.cdiv(self.T,block)
        t=torch.arange(block,device=p.x.device);d=t[None,:]-t[:,None]
        # Use double for coefficient preparation, then cache in FP32.
        a=p.da.double()[:,None,None];c=p.dc.double()[:,None,None]
        h=p.om.double()[:,None,None]*(a**d.clamp_min(0)[None]-c**d.clamp_min(0)[None])
        self.h=torch.where(d[None]>=0,h,0.).float().contiguous()
        self.local=torch.empty((self.B,self.K,self.C*block),device=p.x.device)
        self.ea=torch.empty((self.B,self.K,self.C),device=p.x.device)
        self.ec=torch.empty_like(self.ea);self.sa=torch.empty_like(self.ea);self.sc=torch.empty_like(self.ea)
        self.out=torch.empty((self.B,self.T,self.K),device=p.x.device)

    def __call__(self):
        p=self.p
        self.compiled=filter_local[(self.B*self.K,triton.cdiv(self.C,self.rows))](p.x,self.h,self.local,self.ea,self.ec,
            p.da,p.dc,self.K,self.T,self.C,self.L,self.rows,self.precision,self.rounding,num_warps=4)
        filter_boundaries[(self.B*self.K,)](self.ea,self.ec,self.sa,self.sc,p.da,p.dc,
            self.K,self.C,self.L,triton.next_power_of_2(self.C),num_warps=4,enable_fp_fusion=False)
        filter_join[(triton.cdiv(self.T,32),triton.cdiv(self.K,32),self.B)](self.local,self.sa,self.sc,self.out,
            p.da,p.dc,p.om,self.K,self.T,self.C,self.L,32,32,num_warps=4,enable_fp_fusion=False)
        return self.out
