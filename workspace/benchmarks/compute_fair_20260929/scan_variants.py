"""Exact affine scan alternatives for compact integer-ms impulse rasters.

Input is signed int8 B,K,Tin. STRIDE=2 inserts zero impulses logically;
no half-ms input tensor is materialized. All accumulator arithmetic is FP32.
"""
import torch
import triton
import triton.language as tl
from triton_scan import affine_combine

@triton.jit
def compact_serial(X,Y,DA,DC,OM,K:tl.constexpr,TIN:tl.constexpr,T:tl.constexpr,
                   STRIDE:tl.constexpr,L:tl.constexpr,PAIR:tl.constexpr):
    n=tl.program_id(0);b=tl.program_id(1);ix=tl.arange(0,L)
    da=tl.load(DA+n);dc=tl.load(DC+n);om=tl.load(OM+n);sa=0.;sc=0.
    dda=da*da if PAIR else da;ddc=dc*dc if PAIR else dc
    length=TIN if PAIR else T
    for start in range(tl.cdiv(length,L)):
        t=start*L+ix
        if PAIR:v=tl.load(X+(b*K+n)*TIN+t,t<TIN,0).to(tl.float32)
        else:v=tl.load(X+(b*K+n)*TIN+t//STRIDE,(t<T)&(t%STRIDE==0),0).to(tl.float32)
        aa,xa=tl.associative_scan((tl.full((L,),1.,tl.float32)*dda,v),0,affine_combine)
        cc,xc=tl.associative_scan((tl.full((L,),1.,tl.float32)*ddc,v),0,affine_combine)
        ya=xa+aa*sa;yc=xc+cc*sc
        if PAIR:
            tl.store(Y+(b*T+2*t)*K+n,om*(ya-yc),2*t<T)
            tl.store(Y+(b*T+2*t+1)*K+n,om*(da*ya-dc*yc),2*t+1<T)
        else:tl.store(Y+(b*T+t)*K+n,om*(ya-yc),t<T)
        sa=tl.sum(tl.where(ix==L-1,ya,0.));sc=tl.sum(tl.where(ix==L-1,yc,0.))

@triton.jit
def local_blocks(X,LA,LC,BA,BC,PA,PC,DA,DC,K:tl.constexpr,TIN:tl.constexpr,T:tl.constexpr,
                 STRIDE:tl.constexpr,L:tl.constexpr,NB:tl.constexpr,RECOMPUTE:tl.constexpr):
    n=tl.program_id(0);block=tl.program_id(1);b=tl.program_id(2);i=tl.arange(0,L);t=block*L+i
    da=tl.load(DA+n);dc=tl.load(DC+n)
    v=tl.load(X+(b*K+n)*TIN+t//STRIDE,(t<T)&(t%STRIDE==0),0).to(tl.float32)
    aa,xa=tl.associative_scan((tl.full((L,),1.,tl.float32)*da,v),0,affine_combine)
    cc,xc=tl.associative_scan((tl.full((L,),1.,tl.float32)*dc,v),0,affine_combine)
    tl.store(BA+(b*K+n)*NB+block,tl.sum(tl.where(i==L-1,xa,0.)))
    tl.store(BC+(b*K+n)*NB+block,tl.sum(tl.where(i==L-1,xc,0.)))
    if block==0:
        tl.store(PA+n*L+i,aa);tl.store(PC+n*L+i,cc)
    if not RECOMPUTE:
        tl.store(LA+(b*K+n)*(NB*L)+t,xa)
        tl.store(LC+(b*K+n)*(NB*L)+t,xc)

@triton.jit
def block_prefix(BA,BC,PA,PC,CA,CC,K:tl.constexpr,L:tl.constexpr,NB:tl.constexpr,N:tl.constexpr):
    n=tl.program_id(0);b=tl.program_id(1);j=tl.arange(0,N)
    ap=tl.load(PA+n*L+L-1);cp=tl.load(PC+n*L+L-1)
    a=tl.load(BA+(b*K+n)*NB+j,j<NB,0);c=tl.load(BC+(b*K+n)*NB+j,j<NB,0)
    _,aa=tl.associative_scan((tl.where(j<NB,ap,1.),a),0,affine_combine)
    _,cc=tl.associative_scan((tl.where(j<NB,cp,1.),c),0,affine_combine)
    tl.store(CA+(b*K+n)*NB+j,aa,j<NB);tl.store(CC+(b*K+n)*NB+j,cc,j<NB)

@triton.jit
def finish_stored(LA,LC,PA,PC,CA,CC,OM,Y,K:tl.constexpr,T:tl.constexpr,L:tl.constexpr,
                  NB:tl.constexpr,BT:tl.constexpr,BK:tl.constexpr):
    t=tl.program_id(0)*BT+tl.arange(0,BT);n=tl.program_id(1)*BK+tl.arange(0,BK);b=tl.program_id(2)
    block=t//L;local=t%L;valid=(t[:,None]<T)&(n[None,:]<K)
    ia=(b*K+n[None,:])*(NB*L)+t[:,None]
    la=tl.load(LA+ia,valid,0);lc=tl.load(LC+ia,valid,0)
    ca=tl.load(CA+(b*K+n[None,:])*NB+block[:,None]-1,valid&(block[:,None]>0),0)
    cc=tl.load(CC+(b*K+n[None,:])*NB+block[:,None]-1,valid&(block[:,None]>0),0)
    pa=tl.load(PA+n[None,:]*L+local[:,None],n[None,:]<K,0)
    pc=tl.load(PC+n[None,:]*L+local[:,None],n[None,:]<K,0)
    om=tl.load(OM+n,n<K,0)
    tl.store(Y+(b*T+t[:,None])*K+n[None,:],om[None,:]*((la+pa*ca)-(lc+pc*cc)),valid)

@triton.jit
def finish_recompute(X,DA,DC,CA,CC,OM,Y,K:tl.constexpr,TIN:tl.constexpr,T:tl.constexpr,
                      STRIDE:tl.constexpr,L:tl.constexpr,NB:tl.constexpr):
    n=tl.program_id(0);block=tl.program_id(1);b=tl.program_id(2);i=tl.arange(0,L);t=block*L+i
    da=tl.load(DA+n);dc=tl.load(DC+n);om=tl.load(OM+n)
    ca=tl.load(CA+(b*K+n)*NB+block-1,block>0,0);cc=tl.load(CC+(b*K+n)*NB+block-1,block>0,0)
    v=tl.load(X+(b*K+n)*TIN+t//STRIDE,(t<T)&(t%STRIDE==0),0).to(tl.float32)
    aa,xa=tl.associative_scan((tl.full((L,),1.,tl.float32)*da,v),0,affine_combine)
    pc,xc=tl.associative_scan((tl.full((L,),1.,tl.float32)*dc,v),0,affine_combine)
    tl.store(Y+(b*T+t)*K+n,om*((xa+aa*ca)-(xc+pc*cc)),t<T)

class CompactScan:
    def __init__(self,x,y,da,dc,om,dt,mode='compact',block=512):
        self.x,self.y,self.da,self.dc,self.om=x,y,da,dc,om
        self.B,self.K,self.Tin=x.shape;self.T=y.shape[1];self.stride=round(1/dt)
        assert self.T==(self.Tin-1)*self.stride+1 and x.dtype==torch.int8
        self.mode,self.block=mode,block;self.nb=triton.cdiv(self.T,block)
        if mode.startswith('hier'):
            shape=(self.B,self.K,self.nb)
            self.ba=torch.empty(shape,device=x.device);self.bc=torch.empty_like(self.ba)
            self.ca=torch.empty_like(self.ba);self.cc=torch.empty_like(self.ba)
            self.pa=torch.empty((self.K,block),device=x.device);self.pc=torch.empty_like(self.pa)
            local=(self.B,self.K,self.nb*block) if mode=='hier_store' else (1,)
            self.la=torch.empty(local,device=x.device);self.lc=torch.empty_like(self.la)
    def __call__(self):
        B,K,T,L,NB=self.B,self.K,self.T,self.block,self.nb
        if self.mode in ['compact','pair']:
            compact_serial[(K,B)](self.x,self.y,self.da,self.dc,self.om,K,self.Tin,T,self.stride,L,self.mode=='pair',num_warps=4,enable_fp_fusion=False)
        else:
            rec=self.mode=='hier_recompute'
            local_blocks[(K,NB,B)](self.x,self.la,self.lc,self.ba,self.bc,self.pa,self.pc,self.da,self.dc,K,self.Tin,T,self.stride,L,NB,rec,num_warps=4,enable_fp_fusion=False)
            block_prefix[(K,B)](self.ba,self.bc,self.pa,self.pc,self.ca,self.cc,K,L,NB,triton.next_power_of_2(NB),num_warps=4,enable_fp_fusion=False)
            if rec:
                finish_recompute[(K,NB,B)](self.x,self.da,self.dc,self.ca,self.cc,self.om,self.y,K,self.Tin,T,self.stride,L,NB,num_warps=4,enable_fp_fusion=False)
            else:
                finish_stored[(triton.cdiv(T,32),triton.cdiv(K,32),B)](self.la,self.lc,self.pa,self.pc,self.ca,self.cc,self.om,self.y,K,T,L,NB,32,32,num_warps=4,enable_fp_fusion=False)
        return self.y
