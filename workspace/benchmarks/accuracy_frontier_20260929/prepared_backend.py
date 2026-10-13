"""Prepared inference for one/two temporal components; physical time in ms.

Import with the project's compute_fair_20260929 and sparse_20260928 folders
on sys.path, as in evaluate_benchmark.py. State must be subset to active IDs.
Input is signed int8 [B, A, Tin] with zero initial filter/reset state.
"""
import torch
from backend import Model,Graph
from scan_variants import CompactScan

class MultiKernelModel(Model):
    def __init__(self,state,x,dt,mode='compact',block=512):
        super().__init__(state,x,dt,mode,block)
        self.aux='aux_omega' in state
        if self.aux:
            self.ay=torch.empty_like(self.y)
            self.ada=torch.exp(-dt/state['aux_tau_decay'])
            self.adc=torch.exp(-dt*(1/state['aux_tau_decay']+1/state['aux_tau_rise']))
            self.ascan=CompactScan(x,self.ay,self.ada,self.adc,state['aux_omega'],dt,mode,block)

    def filter(self):
        self.scan()
        if self.aux:
            self.ascan();self.y.add_(self.ay)
        return self.y

def prepare(state,x,output_interval_ms,input_interval_ms=1.,mode='compact',block=512):
    """No change to event amplitudes; taus are physical ms in the checkpoint.

    For native half-ms events pass input_interval_ms=.5 and output_interval_ms=.5.
    For the old integer-ms main31 inputs pass input_interval_ms=1 and output=.5.
    The preparation converts physical taus once into input-grid units. The
    scan's logical stride then inserts only the necessary zero event bins.
    """
    if output_interval_ms not in (1.,.5) or input_interval_ms not in (1.,.5):
        raise ValueError('Validated physical intervals are 1 and 0.5 ms')
    ratio=output_interval_ms/input_interval_ms
    if ratio not in (1.,.5):raise ValueError('Supported logical output/input ratios are 1 and 0.5')
    if mode=='pair' and ratio!=.5:raise ValueError('Pair scan requires two output samples per input sample')
    st={k:v.detach().clone() for k,v in state.items()}
    for key in ['tau_rise','tau_decay','tau_reset','aux_tau_rise','aux_tau_decay']:
        if key in st:st[key]=st[key]/input_interval_ms
    model=MultiKernelModel(st,x,ratio,mode,block)
    model.output_interval_ms=float(output_interval_ms);model.input_interval_ms=float(input_interval_ms)
    return model,Graph(model)
