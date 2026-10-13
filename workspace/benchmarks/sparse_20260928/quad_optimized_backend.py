"""Opt-in integration; frozen original code and existing backends stay intact."""
from backends import Prepared
from quadratic_tiled import TiledDrive


class TiledPrepared(Prepared):
    def __init__(self,state,x,spec,scan='triton'):
        super().__init__(state,x,scan,'cublas')
        self.tiled_drive=TiledDrive(self.y.reshape(-1,self.K),self.W,self.bias,**spec)
        self.drive=self.tiled_drive.out.view(self.B,self.T)
        # The specialized contraction never materializes Z=YW.
        del self.z

    def drive_only(self,y):
        self.tiled_drive(y.reshape(-1,self.K))
        return self.drive


def make_optimized_prepared(state,x,policy):
    """Unknown shapes retain cuBLAS; known entries come from explicit retests."""
    B,A,T=x.shape
    entry=policy.get('shapes',{}).get(f'{A}:{T}:{B}')
    if not entry or not entry.get('enabled'):
        return Prepared(state,x,'triton','cublas')
    return TiledPrepared(state,x,entry['spec'])
