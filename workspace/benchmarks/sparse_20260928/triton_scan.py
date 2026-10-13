import torch
import triton
import triton.language as tl

@triton.jit
def affine_combine(a1,b1,a2,b2):
    return a2*a1, a2*b1+b2

@triton.jit
def parallel_scan(X,Y,DA,DC,OM,K:tl.constexpr,T:tl.constexpr,L:tl.constexpr):
    n=tl.program_id(0); b=tl.program_id(1)
    ix=tl.arange(0,L)
    da=tl.load(DA+n);dc=tl.load(DC+n);om=tl.load(OM+n)
    sa=0.;sc=0.
    for start in range(tl.cdiv(T,L)):
        t=start*L+ix
        v=tl.load(X+(b*K+n)*T+t,t<T,other=0.)
        aa,xa=tl.associative_scan((tl.full((L,),1.,tl.float32)*da,v),0,affine_combine)
        cc,xc=tl.associative_scan((tl.full((L,),1.,tl.float32)*dc,v),0,affine_combine)
        ya=xa+aa*sa;yc=xc+cc*sc
        tl.store(Y+(b*T+t)*K+n,om*(ya-yc),t<T)
        sa=tl.sum(tl.where(ix==L-1,ya,0.))
        sc=tl.sum(tl.where(ix==L-1,yc,0.))

def run_scan(x,y,da,dc,om,block=128):
    B,K,T=x.shape
    parallel_scan[(K,B)](x,y,da,dc,om,K,T,block,num_warps=4,enable_fp_fusion=False)
