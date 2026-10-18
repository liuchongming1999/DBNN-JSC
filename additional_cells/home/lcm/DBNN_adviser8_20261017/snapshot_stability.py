"""Resample validation choices among six archived, individually frozen snapshots."""
from pathlib import Path
import argparse
import hashlib
import numpy as np
import torch
from common8 import read,write,sha,metric,aggregate,match_events,event_aggregate,reset_function
from train_full_matrix import VariableDrive,fft_length,state_digest


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);a=ap.parse_args()
    work=a.work.resolve();root=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
    source=root/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag'
    selection=read(source/'final_selection.json');candidates=selection['candidates'];assert len(candidates)==6
    rows=[r for r in read(source/'training_manifest.json')['cases'] if r['split']=='validation']
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest');reset=reset_function(work)
    inputs=[];targets=[];events=[]
    for r in rows:
        d=np.load(r['input_file']);ids=d['ids'];x=np.zeros((500,6001),np.float32)
        assert hashlib.sha256(d['x'].tobytes()).hexdigest()==r['input_sha256']
        x[ids]=d['x']*np.where(ids[:,None]<250,1,-1)
        inputs.append(torch.fft.rfft(torch.from_numpy(x).cuda()[None],n=fft_length(6001)))
        f=np.load(r['reference_file']);targets.append(np.interp(np.arange(6001),f['time_ms'],f['voltage']))
        events.append(f['spike_times'])
    metrics=[];event_counts=[];records=[]
    for c in candidates:
        assert sha(c['checkpoint'])==c['checkpoint_sha256']
        st=torch.load(c['checkpoint'],map_location='cpu',weights_only=False)['model_state']
        model=VariableDrive(st,torch.device('cuda'),1.,True);p=c['reset'];ms=[];es=[]
        for xf,ref,ev in zip(inputs,targets,events):
            u=model(xf,6001)[0].cpu().numpy();v,pred=reset(u,p['v_th'],p['reset_amp'],p['tau_reset'])
            ms.append(metric(ref,v,ev));es.append(match_events(ev,pred))
        summary=aggregate(ms);e=event_aggregate(es)
        normalized=np.mean([r['sse']/max(r['reference_ss'],.04*r['n']) for r in ms])
        J=summary['pooled_rmse_mv']**2+.02*normalized+.05*(1-(e['f1'] or 0))
        assert abs(J-c['validation']['score'])<1e-6,(c['name'],J,c['validation']['score'])
        records.append(dict(name=c['name'],checkpoint_sha256=c['checkpoint_sha256'],
            physical_state_digest=state_digest(st),reset=p,full_validation_J=J,
            pooled_rmse_mv=summary['pooled_rmse_mv'],events=e))
        metrics.append(ms);event_counts.append(es)
        del model
    rng=np.random.default_rng(20261017);ix=rng.integers(0,32,(10000,32))
    sse=np.array([[m['sse'] for m in ms] for ms in metrics])
    n=np.array([m['n'] for m in metrics[0]])
    normalized=np.array([[m['sse']/max(m['reference_ss'],.04*m['n']) for m in ms] for ms in metrics])
    freq=np.zeros(6,int);all_scores=[]
    for ms,es,ns in zip(sse,event_counts,normalized):
        tp,fp,fn=[np.array([e[k] for e in es])[ix].sum(1) for k in ['tp','fp','fn']]
        den=2*tp+fp+fn;f1=np.divide(2*tp,den,out=np.zeros(10000),where=den>0)
        all_scores.append(ms[ix].sum(1)/n[ix].sum(1)+.02*ns[ix].mean(1)+.05*(1-f1))
    chosen=np.argmin(np.array(all_scores),axis=0)
    for i,r in enumerate(records):r['selection_fraction']=float(np.mean(chosen==i))
    groups={}
    for r in records:
        key=r['physical_state_digest']
        g=groups.setdefault(key,dict(names=[],selection_fraction=0.))
        g['names'].append(r['name']);g['selection_fraction']+=r['selection_fraction']
    np.savez_compressed(work/'frozen_primary/snapshot_validation_arrays.npz',
        sse=sse,n=n,normalized=normalized,
        tp=np.array([[e['tp'] for e in es] for es in event_counts]),
        fp=np.array([[e['fp'] for e in es] for es in event_counts]),
        fn=np.array([[e['fn'] for e in es] for es in event_counts]),
        bootstrap_selected_indices=chosen,case_ids=np.array([r['case_id'] for r in rows]))
    write(work/'frozen_primary/snapshot_stability.json',dict(passed=True,replicates=10000,
        candidates=records,equivalent_snapshot_groups=list(groups.values()),
        final_checkpoint_and_reset_unchanged=True,
        scope='All six archived candidates keep their own original validation-calibrated reset. Resampling changes only the counterfactual ranking, not any model. Duplicate physical states are grouped; exact ties choose the first archived candidate.'))
    print('SNAPSHOT_STABILITY',[(r['name'],r['selection_fraction']) for r in records],flush=True)


if __name__=='__main__':main()
