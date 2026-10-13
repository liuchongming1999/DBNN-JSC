"""Physical-time conventions shared by training, inference and analysis."""
import hashlib, json
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sample_state(state,dt):
    # Checkpoints always store physical milliseconds. Only the backend adapter
    # converts time constants into sample counts, exactly once.
    out={k:v.detach().clone() for k,v in state.items()}
    for k in ['tau_rise','tau_decay','tau_reset']:out[k]=out[k]/dt
    return out
def raster(raw,dt):
    stride=int(round(1/dt));assert abs(stride*dt-1)<1e-9
    x=np.zeros((raw.shape[0],(raw.shape[1]-1)*stride+1),np.float32)
    x[:,::stride]=raw
    assert np.array_equal(x[:,::stride],raw) and float(x.sum())==float(raw.sum())
    return x
def mask(n,spikes,dt):
    t=np.arange(n)*dt;keep=np.ones(n,bool)
    for s in spikes:keep&=np.abs(t-float(s))>10.
    return keep
def metric(ref,pred,spikes,dt):
    keep=mask(len(ref),spikes,dt);r=np.asarray(ref,float)[keep];p=np.asarray(pred,float)[keep];e=p-r
    assert np.isfinite(r).all() and np.isfinite(p).all()
    return dict(n=int(keep.sum()),sse=float(np.square(e).sum()),error_ss=float(((e-e.mean())**2).sum()),
        reference_ss=float(((r-r.mean())**2).sum()),rmse_mv=float(np.sqrt(np.mean(e**2))),
        ve=float(1-e.var()/r.var()),bias_mv=float(e.mean()))
def aggregate(rows):
    return dict(cases=len(rows),pooled_rmse_mv=float(np.sqrt(sum(r['sse'] for r in rows)/sum(r['n'] for r in rows))),
        mean_case_ve=float(np.mean([r['ve'] for r in rows])),
        within_trace_weighted_ve=float(1-sum(r['error_ss'] for r in rows)/sum(r['reference_ss'] for r in rows)))
