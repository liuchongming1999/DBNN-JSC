"""Cached CPU implementation of the locked DBNN model (B=1, zero state)."""
import ctypes as C
from pathlib import Path
import numpy as np
import torch

P=C.c_void_p; I=C.c_int; F=C.c_float

class CPUModel:
    def __init__(self,state,ids,T):
        self.ids=np.ascontiguousarray(ids,dtype=np.int64);self.T=int(T)
        ix=torch.from_numpy(self.ids)
        self.da=torch.exp(-1/state['tau_decay'][ix]).numpy().copy()
        self.dc=torch.exp(-(1/state['tau_rise'][ix]+1/state['tau_decay'][ix])).numpy().copy()
        self.omega=state['omega'][ix].numpy().copy()
        self.w=state['bilinear.weight'][0][ix][:,ix].contiguous().numpy().copy()
        self.bias=float(state['bilinear.bias']);self.amp=abs(float(state['reset_amp']))
        self.threshold=float(state['v_th']);self.alpha=float(torch.exp(-1/state['tau_reset'].abs()))
        self.lib=C.CDLL(str(Path(__file__).with_name('cpu_inference.so')))
        self.mkl=C.CDLL(str(Path(torch.__file__).parent/'lib/libtorch_cpu.so'))
        self.lib.dbnn_cpu_create.argtypes=[I,I,P,P,P,P,P,F,F,F,F,P,P]
        self.lib.dbnn_cpu_create.restype=P
        self.lib.dbnn_cpu_destroy.argtypes=[P]
        self.lib.dbnn_cpu_forward.argtypes=[P,P,P,I,I,I,I,P]
        self.lib.dbnn_cpu_forward.restype=I
        self.lib.dbnn_cpu_copy_drive.argtypes=[P,P]
        self.handle=self.lib.dbnn_cpu_create(len(ids),T,*[a.ctypes.data for a in [self.ids,self.da,self.dc,self.omega,self.w]],
            self.bias,self.amp,self.alpha,self.threshold,C.cast(self.mkl.sgemm_,P),C.cast(self.mkl.MKL_Set_Num_Threads_Local,P))
        self.output=np.empty(T,dtype=np.float32)
        self.events_count=0

    def run(self,dense,threads=1,filter_mode=1,drive_mode=0,reset_mode=1,profile=None):
        assert dense.dtype==np.float32 and dense.flags.c_contiguous and dense.shape[1]==self.T
        self.events_count=self.lib.dbnn_cpu_forward(self.handle,dense.ctypes.data,self.output.ctypes.data,
            threads,filter_mode,drive_mode,reset_mode,None if profile is None else profile.ctypes.data)
        return self.output

    def drive(self):
        out=np.empty(self.T,dtype=np.float32);self.lib.dbnn_cpu_copy_drive(self.handle,out.ctypes.data);return out

    def close(self):
        if self.handle:self.lib.dbnn_cpu_destroy(self.handle);self.handle=None

    def __del__(self):
        if getattr(self,'handle',None):self.close()
