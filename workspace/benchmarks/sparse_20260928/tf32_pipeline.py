"""Opt-in mixed pipeline; only declared matrix products use Tensor Cores."""
from backends import Prepared
from quadratic_tiled import TiledDrive
from tf32_kernels import TensorDrive, TensorFilter
from quad_optimized_backend import make_optimized_prepared


class PrecisionPipeline(Prepared):
    def __init__(self,state,x,drive_spec,filter_spec=None,scan_block=128):
        super().__init__(state,x,'triton','cublas',block=scan_block)
        self.drive_spec=dict(drive_spec);self.filter_spec=filter_spec
        kind=self.drive_spec.pop('kind')
        self.custom_drive=None
        if kind=='ieee':self.custom_drive=TiledDrive(self.y.reshape(-1,self.K),self.W,self.bias,**self.drive_spec)
        elif kind=='tensor':self.custom_drive=TensorDrive(self.y.reshape(-1,self.K),self.W,self.bias,**self.drive_spec)
        elif kind=='cublas':self.quad='tf32' if self.drive_spec.get('tf32') else 'cublas'
        else:raise ValueError(kind)
        if self.custom_drive is not None:
            self.drive=self.custom_drive.out.view(self.B,self.T)
            del self.z
        self.tc_filter=TensorFilter(self,**filter_spec) if filter_spec is not None else None

    def synaptic(self):
        return self.tc_filter() if self.tc_filter is not None else super().synaptic()

    def drive_only(self,y):
        if self.custom_drive is None:return super().drive_only(y)
        self.custom_drive(y.reshape(-1,self.K))
        return self.drive


def make_precision_prepared(state,x,policy,profile='strict'):
    """Explicit inference precision profile; no silent change to IEEE defaults.

    strict: <1e-3 mV on all held-out checks and zero reset-bin differences.
    approximate_0p01mv: opt-in <1e-2 mV and zero reset-bin differences.
    These describe measured validation gates, not bounds for arbitrary inputs.
    """
    B,A,T=x.shape
    entries=policy['profiles'][profile]['shapes']
    entry=entries.get(f'{A}:{T}:{B}')
    if entry is None or not entry.get('enabled'):
        return make_optimized_prepared(state,x,policy['ieee_fallback'])
    spec=entry['spec']
    return PrecisionPipeline(state,x,spec['drive_spec'],spec.get('filter_spec'),spec.get('scan_block',128))
