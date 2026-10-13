"""Prepared FP32 inference; all new scans retain the same model and reset."""
import torch
from backends import native
from triton_scan import run_scan
from quadratic_tiled import TiledDrive
from scan_variants import CompactScan

class Model:
    def __init__(self,state,x,dt,mode='compact',block=512):
        assert x.dtype==torch.int8 and x.is_cuda and x.is_contiguous()
        self.x=x;self.B,self.K,self.Tin=x.shape;self.dt=dt;self.stride=round(1/dt)
        self.T=(self.Tin-1)*self.stride+1;self.mode=mode;self.block=block
        self.da=torch.exp(-dt/state['tau_decay']);self.dc=torch.exp(-dt*(1/state['tau_rise']+1/state['tau_decay']))
        self.om=state['omega'].contiguous();self.W=state['bilinear.weight'][0].contiguous();self.bias=float(state['bilinear.bias'])
        self.alpha=float(torch.exp(-dt/state['tau_reset'].abs()));self.amp=abs(float(state['reset_amp']));self.th=float(state['v_th'])
        self.powers=torch.tensor(self.alpha,device=x.device).pow(torch.arange(self.T,device=x.device,dtype=torch.float32))
        self.y=torch.empty((self.B,self.T,self.K),device=x.device);self.out=torch.empty((self.B,self.T),device=x.device)
        if mode=='baseline':
            self.full=torch.zeros((self.B,self.K,self.T),device=x.device);self.full[...,::self.stride]=x.float()
            self.scan=lambda:run_scan(self.full,self.y,self.da,self.dc,self.om,block)
        else:self.scan=CompactScan(x,self.y,self.da,self.dc,self.om,dt,mode,block)
        self.tile=TiledDrive(self.y.reshape(-1,self.K),self.W,self.bias,'triton',3,self.K>32,'ieee')
        self.drive=self.tile.out.view(self.B,self.T)
    def filter(self):self.scan();return self.y
    def integrate(self):self.tile(self.y.reshape(-1,self.K));return self.drive
    def reset(self):
        native().call('dbnn_reset_event',self.drive,self.out,self.powers,self.B,self.T,self.alpha,self.amp,self.th,torch.cuda.current_stream().cuda_stream)
        return self.out
    def __call__(self):self.filter();self.integrate();return self.reset()

class Graph:
    def __init__(self,fn,events=False):
        self.fn=fn
        for _ in range(3):fn()
        torch.cuda.synchronize();self.g=torch.cuda.CUDAGraph()
        if events:
            self.start=torch.cuda.Event(enable_timing=True,external=True)
            self.end=torch.cuda.Event(enable_timing=True,external=True)
        with torch.cuda.graph(self.g):
            if events:self.start.record()
            self.out=fn()
            if events:self.end.record()
    def __call__(self):self.g.replay();return self.out
    def elapsed_ms(self):return self.start.elapsed_time(self.end)

def numerical(actual,expected,threshold):
    a=actual.detach().float();b=expected.detach().float();e=a-b
    result=dict(max_abs_mv=float(e.abs().max()),rmse_mv=float(e.square().mean().sqrt()),
        events_equal=bool(torch.equal(a>=threshold,b>=threshold)),finite=bool(torch.isfinite(a).all()))
    result['passed']=result['finite'] and result['events_equal'] and result['max_abs_mv']<.001
    return result

def options(dt):
    opts=[('baseline',512),('compact',256),('compact',512),('compact',1024)]
    for mode in ['hier_store','hier_recompute']:
        opts += [(mode,L) for L in [256,512,1024]]
    if dt==.5:opts += [('pair',L) for L in [256,512,1024]]
    return opts

def make_prepared(state,x,dt,policy):
    """State is already restricted to active IDs; taus remain in physical ms.

    x: contiguous CUDA int8 [B,A,Tin], signed integer-ms event amplitudes.
    Copy new values into the same x buffer before replaying the returned graph.
    The measured workload is B=1 and zero initial state. Preparation is untimed.
    """
    import math
    if dt not in (1.,.5):raise ValueError('Validated output intervals are 1 and 0.5 ms')
    candidates=[q for q in policy['records'] if q['A']==x.shape[1] and q['dt_ms']==dt]
    if candidates:
        near=min(candidates,key=lambda q:abs(math.log(q['input_steps']/x.shape[2])))
        valid=[n for n in near['timing'] if not n.startswith('baseline') and near['validation'][n]['passed']]
        selected=min(valid,key=lambda n:near['timing'][n]['device_ms'])
        rule=dict(selected=selected,calibration_input_steps=near['input_steps'],fallback=False)
    else:
        selected='compact_512';rule=dict(selected=selected,calibration_input_steps=None,fallback=True)
    mode,block=selected.rsplit('_',1);model=Model(state,x,dt,mode,int(block))
    return model,Graph(model),rule
