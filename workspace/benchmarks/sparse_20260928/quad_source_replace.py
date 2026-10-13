"""Insert the new kernel at the original F/R drive call sites only."""
import json,os
import torch
from experiment import HERE,DBNN_F,DBNN_R,accuracy,measure_group,stamp,resource_snapshot
from quad_tune import make_data
from original_baseline_audit import attach,variant_source
from quadratic_tiled import TiledDrive
from quadratic_benchmark import save

class OriginalDrive:
    def __init__(self,B,A,T,chunk,w,bias,spec):
        self.fns={}
        for length in {min(chunk,T),T%chunk}-{0}:
            self.fns[length]=TiledDrive(torch.empty(B*length,A,device=w.device),w,bias,**spec)
    def __call__(self,y):
        B,L,A=y.shape
        return self.fns[L](y.contiguous().reshape(B*L,A)).view(B,L)

torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
data=json.loads((HERE/'quad_confirm.json').read_text());assert data.get('finished')
result=dict(started=stamp(),baseline_sha256=data['baseline_sha256'],physical_gpu=os.getenv('CUDA_VISIBLE_DEVICES'),
            scope='original F/R eager forward vs only its full drive replaced; R chunk=256 retained',cases=[])
with torch.no_grad():
    for c in data['cases']:
        if c['A']!=500:continue
        A,T,B=c['A'],c['T'],c['B'];spec=c['spec'];p,_,w=make_data(A,T,B)
        fns={};owners=[]
        for name,cls in [('F',DBNN_F),('R',DBNN_R)]:
            for new in [False,True]:
                m=cls(A,T,torch.device('cuda'));m.load_state_dict(p.source_state);m.eval()
                if new:
                    attach(m,variant_source(cls,'drive'))
                    m._audit_drive=OriginalDrive(B,A,T,256 if name=='R' else T,w,p.bias,spec)
                key=f'{"new" if new else "original"}_{name}';fns[key]=lambda m=m:m(p.x);owners.append(m)
        ref=fns['original_R']().clone()
        errs={k:accuracy(fn().clone(),ref,p.vth) for k,fn in fns.items()}
        for er in errs.values():assert er['finite'] and er['max_abs_mv']<1e-3 and er['spike_mismatch_bins']==0
        for fn in fns.values():
            for _ in range(3):fn()
        record=dict(A=A,T=T,B=B,spec=spec,errors=errs,wall=measure_group(fns,30),resources=resource_snapshot())
        result['cases'].append(record)
        save(HERE/'quad_source_replace.json',result)
        print(B,{k:round(v['median_ms'],4) for k,v in record['wall'].items()},flush=True)
result['finished']=stamp();save(HERE/'quad_source_replace.json',result)
