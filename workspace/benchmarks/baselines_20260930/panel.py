"""Matched event panels, with physical time and the original train/val split."""
import hashlib,json
from pathlib import Path
import numpy as np

MANIFESTS={1.0:'benchmarks/accuracy_frontier_20260929/balanced_1ms/training_manifest.json',.5:'benchmarks/accuracy_frontier_20260929/b2_warm_0p5ms/training_manifest.json',.25:'benchmarks/resolution_frontier_20260929/train/manifest.json'}
EXPECTED={1.0:(117,32),.5:(189,56),.25:(261,80)}

def rows_for(root,dt):
    path=Path(root)/MANIFESTS[dt];rows=json.loads(path.read_text())['cases']
    assert (sum(r['split']=='fit' for r in rows),sum(r['split']=='validation' for r in rows))==EXPECTED[dt]
    return rows,path

def make_case(row,dt,dend=None,signed=False):
    d=np.load(row['input_file']);ids=d['ids'];raw=d['x'];assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256']
    stride=round(float(row['input_interval_ms'])/dt);assert stride>=1 and abs(stride*dt-row['input_interval_ms'])<1e-9
    n=round(row['duration_ms']/dt)+1;x=np.zeros((n,500),np.float32)
    x[np.ix_(np.arange(raw.shape[1])*stride,ids)]=raw.T
    assert float(x.sum())==float(raw.sum())
    if signed:x[:,250:]*=-1
    r=np.load(row['reference_file']);t=np.arange(n)*dt;v=np.interp(t,r['time_ms'],r['voltage']).astype(np.float32)
    sp=np.zeros(n,np.float32)
    for s in r['spike_times']:
        k=int(np.clip(np.floor(float(s)/dt+.5),0,n-1));sp[k]=1
    y=np.stack([sp,np.minimum(v,-55)+67.7],1)
    if signed:y[:,1]*=.1
    aux=np.load(Path(dend)/f"{row['case_id']}.npy",mmap_mode='r') if dend else None
    if aux is not None:assert aux.shape==(n,20)
    return dict(case_id=row['case_id'],split=row['split'],x=x,y=y,raw_voltage=v,spike_times=r['spike_times'],aux=aux)

def sample_batch(cases,window,batch,rng,dt):
    inds=rng.integers(0,len(cases),batch);xs=[];ys=[];ds=[]
    # Authors choose window ends after max(500 ms, window length), with no padding.
    for k in inds:
        c=cases[int(k)];end=int(rng.integers(max(window,round(500/dt)),len(c['x'])))
        xs.append(c['x'][end-window:end]);ys.append(c['y'][end-window:end])
        if c['aux'] is not None:ds.append(np.clip(c['aux'][end-window:end],-3,3))
    return np.stack(xs),np.stack(ys),np.stack(ds) if ds else None

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n');temp.replace(path)
