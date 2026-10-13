"""Original DBNN at 1 / 0.5 / 0.25 ms; exact coarse-event affine scan option."""
import torch,triton
import triton.language as tl
from backend import Model,Graph
from triton_scan import affine_combine

@triton.jit
def event_grid_scan(X,Y,DA,DC,OM,K:tl.constexpr,TIN:tl.constexpr,T:tl.constexpr,S:tl.constexpr,L:tl.constexpr):
    n=tl.program_id(0);b=tl.program_id(1);ix=tl.arange(0,L)
    da=tl.load(DA+n);dc=tl.load(DC+n);om=tl.load(OM+n);sa=0.;sc=0.
    dda=1.;ddc=1.
    for _ in tl.static_range(S):dda*=da;ddc*=dc
    for start in range(tl.cdiv(TIN,L)):
        t=start*L+ix;v=tl.load(X+(b*K+n)*TIN+t,t<TIN,0).to(tl.float32)
        aa,xa=tl.associative_scan((tl.full((L,),1.,tl.float32)*dda,v),0,affine_combine)
        cc,xc=tl.associative_scan((tl.full((L,),1.,tl.float32)*ddc,v),0,affine_combine)
        ya=xa+aa*sa;yc=xc+cc*sc;pa=1.;pc=1.
        for j in tl.static_range(S):
            tl.store(Y+(b*T+S*t+j)*K+n,om*(pa*ya-pc*yc),S*t+j<T)
            pa*=da;pc*=dc
        sa=tl.sum(tl.where(ix==L-1,ya,0.));sc=tl.sum(tl.where(ix==L-1,yc,0.))

class ResolutionModel(Model):
    def __init__(self,state,x,output_dt,mode='compact',block=512,input_dt=1.):
        assert output_dt in (1.,.5,.25) and input_dt in (1.,.5,.25)
        ratio=output_dt/input_dt;assert ratio in (1.,.5,.25)
        assert not any(k.startswith('aux_') for k in state)
        st={k:v.detach().clone() for k,v in state.items()}
        for k in ['tau_rise','tau_decay','tau_reset']:st[k]=st[k]/input_dt
        super().__init__(st,x,ratio,'compact' if mode=='event_grid' else mode,block)
        if mode=='event_grid':
            self.scan=lambda:event_grid_scan[(self.K,self.B)](self.x,self.y,self.da,self.dc,self.om,self.K,self.Tin,self.T,self.stride,block)
        self.output_interval_ms=float(output_dt);self.input_interval_ms=float(input_dt)

def options(dt):
    opts=[('compact',b) for b in [256,512,1024]]
    opts += [(mode,b) for mode in ['hier_store','hier_recompute'] for b in [256,512,1024]]
    if dt==.5:opts += [('pair',b) for b in [256,512,1024]]
    if dt<1:opts += [('event_grid',b) for b in [256,512,1024]]
    return opts
