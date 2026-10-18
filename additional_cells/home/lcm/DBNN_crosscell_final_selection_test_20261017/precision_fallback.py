"""Prepared FFT drive + strictly sequential FP64 reset for numerical parity only.

The learned parameters stay FP32. Reset uses the same FP64 recurrence as the
selection/oracle implementation; no thresholds or output events are adjusted.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _reset64(DRIVE,OUT,PARAMS,N:tl.constexpr):
    alpha=tl.load(PARAMS)
    amp=tl.load(PARAMS+1)
    threshold=tl.load(PARAMS+2)
    r=tl.full((),0.,tl.float64)
    for i in tl.range(0,N,loop_unroll_factor=1):
        voltage=tl.load(DRIVE+i).to(tl.float64)+r
        tl.store(OUT+i,voltage)
        fired=voltage>=threshold
        r=alpha*r-tl.where(fired,amp,0.)


class FFT64ResetModel:
    def __init__(self,state,x):
        self.x=x;self.n=x.shape[-1];self.nfft=1<<(2*self.n-2).bit_length()
        age=torch.arange(self.n,device=x.device)[None]
        k=state['omega'][:,None]*(1-torch.exp(-age/state['tau_rise'][:,None]))*torch.exp(-age/state['tau_decay'][:,None])
        self.kfft=torch.fft.rfft(k,n=self.nfft)[None]
        self.weight=state['bilinear.weight'][0];self.bias=state['bilinear.bias']
        self.th=float(state['v_th']);self.out=torch.empty((1,self.n),device=x.device,dtype=torch.float64)
        self.params=torch.tensor([math.exp(-1/float(state['tau_reset'])),float(state['reset_amp']),self.th],device=x.device,dtype=torch.float64)

    def __call__(self):
        y=torch.fft.irfft(torch.fft.rfft(self.x.float(),n=self.nfft)*self.kfft,n=self.nfft)[...,:self.n].transpose(1,2).contiguous()
        self.drive=((y@self.weight)*y).sum(-1)+y.sum(-1)+self.bias
        _reset64[(1,)](self.drive,self.out,self.params,self.n,num_warps=1,enable_fp_fusion=False)
        return self.out
