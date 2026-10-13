import json,sys,time,traceback
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'source'))
from DBNN import DBNN_R
from backends import Prepared,capture
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False
d=np.load(ROOT/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True)
signs=np.where(d['channel_type'].astype(str)=='exc',1.,-1.).astype(np.float32)
xc=torch.from_numpy(d['test_ip'][:1,:,:1001].astype(np.float32)*signs[None,:,None])
state=torch.load(ROOT/'source/dbnn_checkpoint.pt',map_location='cpu',weights_only=False)['model_state']
r=DBNN_R(500,1001,torch.device('cpu'));r.load_state_dict(state);r.eval()
with torch.inference_mode():ref=r(xc)
dev='cuda';x=xc.to(dev);s={k:v.to(dev) for k,v in state.items()}
recs=[]
for scan,quad in [('cuda','cublas'),('triton','cublas'),('fft','cublas'),('fft','cutlass'),('fft','direct'),('cuda','direct'),('cpu','cublas'),('cute','cublas')]:
    try:
        p=Prepared(state if scan=='cpu' else s,xc if scan=='cpu' else x,scan,quad)
        with torch.inference_mode():o=p().cpu()
        e=o-ref
        if scan!='cpu':fn=capture(p)
        else:fn=p
        with torch.inference_mode():
            original=p.x.clone();p.x.zero_();zero_out=fn().clone().cpu()
            if (zero_out-p.bias).abs().max()>1e-5:raise AssertionError('Graph stale-output check failed')
            p.x.copy_(original);restored=fn().clone().cpu()
            if (restored-o).abs().max()>1e-5:raise AssertionError('Graph restore check failed')
        ts=[]
        for _ in range(10):
            if scan!='cpu':torch.cuda.synchronize()
            t=time.perf_counter();fn()
            if scan!='cpu':torch.cuda.synchronize()
            ts.append((time.perf_counter()-t)*1000)
        rec={'scan':scan,'quad':quad,'rmse':float((e.square().mean()).sqrt()),'max':float(e.abs().max()),'median_ms':float(np.median(ts))}
    except Exception as err:
        traceback.print_exc();rec={'scan':scan,'quad':quad,'error':str(err)}
    recs.append(rec); print(json.dumps(rec),flush=True)
(Path(__file__).parent/'smoke_results.json').write_text(json.dumps(recs,indent=2))
