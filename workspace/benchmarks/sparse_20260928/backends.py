"""Inference backends. Packing an identically-zero channel is an exact reduction.

Inputs are B,K,T. All states start at zero; no per-timestep pruning is applied.
The deployed checkpoint is frozen: coefficients, W and FFT kernels are cached.
"""
from pathlib import Path
import ctypes as C
import math
import torch
from triton_scan import run_scan

HERE=Path(__file__).resolve().parent
P=C.c_void_p; I=C.c_int; F=C.c_float

class Native:
    def __init__(self):
        self.gpu=C.CDLL(str(HERE/'native_kernels_v2.so'))
        self.cpu=C.CDLL(str(HERE/'native_cpu.so'))
        self.gpu.dbnn_scan.argtypes=[P,P,P,P,P,I,I,I,P]
        self.gpu.dbnn_reduce.argtypes=[P,P,P,I,I,F,P]
        self.gpu.dbnn_direct.argtypes=[P,P,P,I,I,F,P]
        self.gpu.dbnn_reset.argtypes=[P,P,I,I,F,F,F,P]
        self.gpu.dbnn_reset_event.argtypes=[P,P,P,I,I,F,F,F,P]
        self.gpu.dbnn_cutlass.argtypes=[P,P,P,I,I,P]
        self.cpu.dbnn_cpu_scan.argtypes=[P,P,P,P,P,I,I,I,I]
        self.cpu.dbnn_cpu_reset.argtypes=[P,P,I,I,F,F,F,I]
    def call(self,name,*args):
        args=[a.data_ptr() if isinstance(a,torch.Tensor) else a for a in args]
        err=getattr(self.gpu,name)(*args)
        if err:raise RuntimeError(f'{name}: CUDA/CUTLASS status {err}')

_native=None
def native():
    global _native
    if _native is None:_native=Native()
    return _native

class Prepared:
    def __init__(self,state,x,scan='fft',quad='cublas',block=128):
        if not x.is_contiguous():
            raise ValueError('Prepared requires contiguous B,K,T input; call contiguous() when packing active sites.')
        self.x=x; self.B,self.K,self.T=x.shape
        self.scan=scan;self.quad=quad;self.block=block
        self.device=x.device; self.threads=torch.get_num_threads()
        self.da=torch.exp(-1/state['tau_decay']).detach().contiguous()
        self.dc=torch.exp(-(1/state['tau_rise']+1/state['tau_decay'])).detach().contiguous()
        self.om=state['omega'].detach().contiguous()
        self.W=state['bilinear.weight'][0].detach().clone().contiguous()
        self.W.fill_diagonal_(0)
        self.bias=float(state['bilinear.bias'].item())
        self.vth=float(state['v_th'].item());self.amp=abs(float(state['reset_amp'].item()))
        self.alpha=float(torch.exp(-1/torch.clamp(state['tau_reset'].abs(),min=1e-4)).item())
        self.alpha_pows=torch.tensor(self.alpha,device=x.device).pow(torch.arange(self.T,device=x.device,dtype=torch.float32))
        self.y=torch.empty((self.B,self.T,self.K),device=x.device)
        self.z=torch.empty_like(self.y)
        self.drive=torch.empty((self.B,self.T),device=x.device)
        self.out=torch.empty_like(self.drive)
        if scan=='fft':
            self.fft_n=1<<(2*self.T-2).bit_length()
            t=torch.arange(self.T,device=x.device)
            kernel=self.om[:,None]*(1-torch.exp(-t/state['tau_rise'][:,None]))*torch.exp(-t/state['tau_decay'][:,None])
            self.kfft=torch.fft.rfft(kernel,n=self.fft_n)
        if x.is_cuda:self.stream=torch.cuda.current_stream(x.device).cuda_stream
        if scan=='cute':
            from cute_backend import make_scan
            self.cute_scan=make_scan(self.x,self.y,self.da,self.dc,self.om)
        if quad in ('bf16','fp16'):
            self.Wlow=self.W.to(torch.bfloat16 if quad=='bf16' else torch.float16)
        if quad=='cutlass_tensor':
            self.tensor_lib=C.CDLL(str(HERE/'cutlass_tensor.so'))
            self.tensor_lib.dbnn_cutlass_tensor.argtypes=[P,P,P,I,I,P]

    def synaptic(self):
        if self.scan=='fft':
            yf=torch.fft.rfft(self.x,n=self.fft_n)*self.kfft
            y=torch.fft.irfft(yf,n=self.fft_n)[...,:self.T]
            return y.transpose(1,2).contiguous()
        if self.scan=='triton':
            run_scan(self.x,self.y,self.da,self.dc,self.om,self.block)
        elif self.scan=='cuda':
            native().call('dbnn_scan',self.x,self.y,self.da,self.dc,self.om,self.B,self.K,self.T,self.stream)
        elif self.scan=='cute':
            self.cute_scan()
        elif self.scan=='cpu':
            native().cpu.dbnn_cpu_scan(*[q.data_ptr() for q in (self.x,self.y,self.da,self.dc,self.om)],self.B,self.K,self.T,self.threads)
        else: raise ValueError(self.scan)
        return self.y

    def drive_only(self,y):
        if self.quad=='direct':
            native().call('dbnn_direct',y,self.W,self.drive,self.B*self.T,self.K,self.bias,self.stream)
        else:
            if self.quad=='cutlass':
                native().call('dbnn_cutlass',y,self.W,self.z,self.B*self.T,self.K,self.stream)
                z=self.z
            elif self.quad=='cutlass_tensor':
                err=self.tensor_lib.dbnn_cutlass_tensor(y.data_ptr(),self.W.data_ptr(),self.z.data_ptr(),self.B*self.T,self.K,self.stream)
                if err:raise RuntimeError(f'CUTLASS TensorOp status {err}')
                z=self.z
            elif self.quad in ('bf16','fp16'):
                yl=y.to(self.Wlow.dtype)
                z=(yl@self.Wlow).float()
            elif self.quad=='tf32':
                old=torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32=True
                torch.mm(y.reshape(-1,self.K),self.W,out=self.z.reshape(-1,self.K))
                torch.backends.cuda.matmul.allow_tf32=old
                z=self.z
            else:
                torch.mm(y.reshape(-1,self.K),self.W,out=self.z.reshape(-1,self.K))
                z=self.z
            if y.is_cuda:
                native().call('dbnn_reduce',y,z,self.drive,self.B*self.T,self.K,self.bias,self.stream)
            else:
                self.drive.copy_((y*z).sum(-1)+y.sum(-1)+self.bias)
        return self.drive

    @torch.inference_mode()
    def __call__(self):
        if self.x.is_cuda:self.stream=torch.cuda.current_stream(self.x.device).cuda_stream
        y=self.synaptic();self.drive_only(y)
        if self.x.is_cuda:
            native().call('dbnn_reset_event',self.drive,self.out,self.alpha_pows,self.B,self.T,self.alpha,self.amp,self.vth,self.stream)
        else:
            native().cpu.dbnn_cpu_reset(self.drive.data_ptr(),self.out.data_ptr(),self.B,self.T,self.alpha,self.amp,self.vth,self.threads)
        return self.out

def capture(prepared):
    for _ in range(3):prepared()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out=prepared()
    # Keep all input/output/coefficient storage alive with the callable.
    def replay():
        graph.replay()
        return out
    replay.prepared=prepared;replay.graph=graph;replay.out=out
    return replay

def subset_state(state,ids):
    out={k:v.detach().clone() for k,v in state.items()}
    for name in ['tau_rise','tau_decay','omega']:out[name]=state[name][ids].contiguous()
    out['bilinear.weight']=state['bilinear.weight'][:,ids][:,:,ids].contiguous()
    return out
