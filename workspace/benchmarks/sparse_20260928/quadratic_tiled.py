"""Tiled complete quadratic form. FP32 IEEE and TF32x3 are explicit modes."""
from pathlib import Path
import ctypes as C
import torch
import triton
import triton.language as tl


@triton.jit
def _contract(Y,W,P,O,M:tl.constexpr,A:tl.constexpr,NT:tl.constexpr,BIAS:tl.constexpr,
              BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,GROUP:tl.constexpr,
              TRI:tl.constexpr,PREC:tl.constexpr):
    pid=tl.program_id(0)
    n_m=tl.cdiv(M,BM)
    group=pid//(GROUP*NT)
    first=group*GROUP
    group_m=tl.minimum(n_m-first,GROUP)
    pm=first+(pid%(GROUP*NT))%group_m
    pn=(pid%(GROUP*NT))//group_m
    rm=pm*BM+tl.arange(0,BM)
    rn=pn*BN+tl.arange(0,BN)
    rk=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    kend=A
    if TRI:kend=tl.minimum(A,(pn+1)*BN)
    for base in range(0,kend,BK):
        k=base+rk
        y=tl.load(Y+rm[:,None]*A+k[None,:],(rm[:,None]<M)&(k[None,:]<A),0)
        w=tl.load(W+k[:,None]*A+rn[None,:],(k[:,None]<A)&(rn[None,:]<A),0)
        acc=tl.dot(y,w,acc,input_precision=PREC)
    yn=tl.load(Y+rm[:,None]*A+rn[None,:],(rm[:,None]<M)&(rn[None,:]<A),0)
    v=tl.sum(acc*yn+yn,1)
    if NT==1:
        tl.store(O+rm,v+BIAS,rm<M)
    else:
        tl.store(P+rm*NT+pn,v,rm<M)


@triton.jit
def _finish(P,O,M:tl.constexpr,NT:tl.constexpr,BIAS:tl.constexpr,NP:tl.constexpr,ROWS:tl.constexpr):
    r=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
    n=tl.arange(0,NP)
    vals=tl.load(P+r[:,None]*NT+n[None,:],(r[:,None]<M)&(n[None,:]<NT),0)
    tl.store(O+r,tl.sum(vals,1)+BIAS,r<M)


@triton.jit
def _pair_contract(Y,W,I,J,P,M:tl.constexpr,A:tl.constexpr,PAIRS:tl.constexpr,
                   BM:tl.constexpr,S:tl.constexpr,GROUP:tl.constexpr,PREC:tl.constexpr):
    pid=tl.program_id(0)
    nm=tl.cdiv(M,BM)
    group=pid//(GROUP*PAIRS)
    first=group*GROUP
    gm=tl.minimum(nm-first,GROUP)
    pm=first+(pid%(GROUP*PAIRS))%gm
    pair=(pid%(GROUP*PAIRS))//gm
    ti=tl.load(I+pair);tj=tl.load(J+pair)
    r=pm*BM+tl.arange(0,BM)
    ki=ti*S+tl.arange(0,S);nj=tj*S+tl.arange(0,S)
    a=tl.load(Y+r[:,None]*A+ki[None,:],(r[:,None]<M)&(ki[None,:]<A),0)
    w=tl.load(W+ki[:,None]*A+nj[None,:],(ki[:,None]<A)&(nj[None,:]<A),0)
    z=tl.dot(a,w,input_precision=PREC)
    yn=tl.load(Y+r[:,None]*A+nj[None,:],(r[:,None]<M)&(nj[None,:]<A),0)
    val=tl.sum(z*yn,1)
    if ti==tj:val+=tl.sum(yn,1)
    tl.store(P+pair*M+r,val,r<M)


@triton.jit
def _finish_columns(P,O,M:tl.constexpr,NT:tl.constexpr,BIAS:tl.constexpr,NP:tl.constexpr,ROWS:tl.constexpr):
    r=tl.program_id(0)*ROWS+tl.arange(0,ROWS)
    n=tl.arange(0,NP)
    v=tl.load(P+n[:,None]*M+r[None,:],(n[:,None]<NT)&(r[None,:]<M),0)
    tl.store(O+r,tl.sum(v,0)+BIAS,r<M)


TRITON_TILES=[(16,64,32,4,2),(32,64,32,4,2),(32,128,32,4,2),
              (64,64,32,4,2),(64,128,32,4,2),(32,64,64,4,3),
              (64,64,64,4,3),(32,128,64,4,3),(64,128,64,8,3),
              (128,64,32,8,3),(16,128,32,4,2),(32,256,32,8,2)]
PAIR_TILES=[(32,32,4),(32,64,4),(64,64,4),(32,128,4),(64,128,8),(16,64,4)]


def upper_weights(w):
    # Includes diagonal once: valid for arbitrary W, not only zero-diagonal W.
    result=torch.triu(w+w.T,diagonal=1)
    result.diagonal().copy_(w.diagonal())
    return result.contiguous()


class TiledDrive:
    def __init__(self,y,w,bias,backend='triton',config=1,triangle=False,precision='ieee'):
        assert y.is_cuda and y.dtype==w.dtype==torch.float32 and y.is_contiguous() and w.is_contiguous()
        self.y,self.w,self.bias=y,w,float(bias)
        self.M,self.A=y.shape
        self.backend,self.config,self.triangle,self.precision=backend,config,triangle,precision
        self.weight=upper_weights(w) if triangle or backend in ('thread','pair') else w
        self.out=torch.empty(self.M,device=y.device,dtype=y.dtype)
        if backend=='triton':bn=TRITON_TILES[config][1]
        elif backend=='cuda':bn=128 if config in (4,5,7) else 64
        else:bn=self.A
        self.partial=torch.empty((self.M,triton.cdiv(self.A,bn)),device=y.device,dtype=y.dtype)
        if backend not in ('triton','pair'):
            self.lib=C.CDLL(str(Path(__file__).with_name('quadratic_tiled_native.so')))
            self.lib.dbnn_tiled.argtypes=[C.c_void_p]*4+[C.c_int]*2+[C.c_float]+[C.c_int]*2+[C.c_void_p]
            for name in ['dbnn_thread','dbnn_direct_fma']:
                getattr(self.lib,name).argtypes=[C.c_void_p]*3+[C.c_int]*2+[C.c_float,C.c_void_p]
            self.lib.dbnn_warp.argtypes=[C.c_void_p]*3+[C.c_int]*2+[C.c_float,C.c_int,C.c_void_p]
        if backend=='thread':self.yt=y.T.contiguous()
        if backend=='pair':
            _,tile,_=PAIR_TILES[config]
            nt=triton.cdiv(self.A,tile)
            indices=[(i,j) for j in range(nt) for i in range(j+1)]
            self.pairs=len(indices)
            self.pair_i=torch.tensor([i for i,j in indices],device=y.device,dtype=torch.int32)
            self.pair_j=torch.tensor([j for i,j in indices],device=y.device,dtype=torch.int32)
            self.partial_cols=torch.empty((self.pairs,self.M),device=y.device)

    def __call__(self,y=None):
        if y is None:y=self.y
        # This wrapper only accepts the row-contiguous layout used by the scan
        # backend. Original R/F wrappers include their required layout conversion.
        assert y.is_contiguous() and y.shape==(self.M,self.A)
        if self.backend=='pair':
            bm,tile,nw=PAIR_TILES[self.config]
            _pair_contract[(triton.cdiv(self.M,bm)*self.pairs,)](y,self.weight,self.pair_i,self.pair_j,
                self.partial_cols,self.M,self.A,self.pairs,bm,tile,8,self.precision,num_warps=nw)
            _finish_columns[(triton.cdiv(self.M,32),)](self.partial_cols,self.out,self.M,self.pairs,self.bias,
                triton.next_power_of_2(self.pairs),32,num_warps=4 if self.pairs<=256 else 8)
        elif self.backend=='triton':
            bm,bn,bk,nw,stages=TRITON_TILES[self.config]
            nt=triton.cdiv(self.A,bn)
            _contract[(triton.cdiv(self.M,bm)*nt,)](y,self.weight,self.partial,self.out,
                self.M,self.A,nt,self.bias,bm,bn,bk,8,self.triangle,self.precision,
                num_warps=nw,num_stages=stages)
            if nt>1:
                _finish[(triton.cdiv(self.M,128),)](self.partial,self.out,self.M,nt,self.bias,
                    triton.next_power_of_2(nt),128,num_warps=4)
        else:
            stream=torch.cuda.current_stream().cuda_stream
            if self.backend=='cuda':
                err=self.lib.dbnn_tiled(y.data_ptr(),self.weight.data_ptr(),self.partial.data_ptr(),self.out.data_ptr(),
                    self.M,self.A,self.bias,self.config,int(self.triangle),stream)
            elif self.backend=='thread':
                # Include changed-input transposition in the timed call.
                self.yt.copy_(y.T)
                err=self.lib.dbnn_thread(self.yt.data_ptr(),self.weight.data_ptr(),self.out.data_ptr(),
                    self.M,self.A,self.bias,stream)
            elif self.backend=='direct_fma':
                err=self.lib.dbnn_direct_fma(y.data_ptr(),self.w.data_ptr(),self.out.data_ptr(),
                    self.M,self.A,self.bias,stream)
            elif self.backend=='warp':
                err=self.lib.dbnn_warp(y.data_ptr(),self.w.data_ptr(),self.out.data_ptr(),
                    self.M,self.A,self.bias,self.config,stream)
            else:raise ValueError(self.backend)
            if err:raise RuntimeError(f'{self.backend}: CUDA error {err}')
        return self.out
