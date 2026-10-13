"""Frozen full-matrix DBNN on all independent confirmatory inputs.

Low-density classification uses the NEURON reference only. Both the full pool
and the prespecified low-density subset are reported, with case bootstrap CIs.
"""
import argparse
import hashlib
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
import torch
from common_panel import read,write,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve()
    manifest=read(work/'confirmatory/manifest.json')
    ref_complete=read(work/'confirmatory/reference_complete.json');assert ref_complete['finished']
    if (work/'evaluation.json').exists():print('ALREADY_EVALUATED');return
    checkpoint=Path(manifest['checkpoint']);assert sha(checkpoint)==manifest['checkpoint_sha256']
    sys.path[:0]=[str(root/'benchmarks/diagonal_1ms_20261009'),str(root/'benchmarks/accuracy_frontier_20260929'),
                 str(root/'benchmarks/training_refinement_20260929')]
    from train_dbnn_diagonal import VariableDrive,fft_length,load_reset
    from common import metric,aggregate
    from train_refinement import match_events,event_aggregate
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state'];device=torch.device('cuda:0')
    model=VariableDrive(state,device,1.,True)
    assert torch.diagonal(state['bilinear.weight'][0]).abs().max()>0
    reset=load_reset(work);pp={k:float(state[k]) for k in ('v_th','reset_amp','tau_reset')}
    records=[];folder=work/'predictions';folder.mkdir(exist_ok=True)
    for row in manifest['cases']:
        inp=np.load(row['input_file']);ids,raw=inp['ids'],inp['x']
        assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256']
        n=row['duration_ms']+1;dense=np.zeros((500,n),np.float32)
        dense[ids]=raw*np.where(ids[:,None]<250,1,-1)
        with torch.inference_mode():
            xf=torch.fft.rfft(torch.from_numpy(dense).to(device)[None],n=fft_length(n))
            drive=model(xf,n)[0].cpu().numpy()
        voltage,events=reset(drive,pp['v_th'],pp['reset_amp'],pp['tau_reset'])
        ref=np.load(work/'confirmatory/references'/f"{row['case_id']}_reference.npz")
        rr=read(work/'confirmatory/references'/f"{row['case_id']}_reference.json")
        mm=metric(ref['voltage'],voltage,ref['spike_times'],1.)
        ee=match_events(ref['spike_times'],events)
        rec=dict(**row,reference_spikes=rr['spike_count'],minimum_isi_ms=rr['minimum_isi_ms'],low_density=rr['low_density'],
                 voltage=mm,events=ee,reference_events_ms=ref['spike_times'].tolist(),predicted_events_ms=events.tolist())
        records.append(rec)
        np.savez_compressed(folder/f"{row['case_id']}.npz",voltage=voltage,drive=drive,time_ms=ref['time_ms'],
                            reference_voltage=ref['voltage'],reference_events_ms=ref['spike_times'],predicted_events_ms=events)
        if len(records)%20==0:
            write(work/'evaluation_status.json',dict(completed=len(records),total=len(manifest['cases'])))
            print('EVALUATE',len(records),len(manifest['cases']),flush=True)

    def summarize(recs):
        if not recs:return None
        result=aggregate([r['voltage'] for r in recs]);result['events']=event_aggregate([r['events'] for r in recs])
        result['reference_events']=sum(r['reference_spikes'] for r in recs)
        result['predicted_events']=sum(len(r['predicted_events_ms']) for r in recs)
        result['total_duration_s']=sum(r['duration_s'] for r in recs)
        result['false_positives_per_minute']=result['events']['fp']/(result['total_duration_s']/60)
        stochastic=[r for r in recs if r['family']!='zero']
        deterministic=[r for r in recs if r['family']=='zero']
        counts=np.array([[r['events']['tp'],r['events']['fp'],r['events']['fn']] for r in stochastic],dtype=np.int64).reshape(-1,3)
        fixed=np.array([[r['events']['tp'],r['events']['fp'],r['events']['fn']] for r in deterministic],dtype=np.int64).reshape(-1,3).sum(0)
        rng=np.random.default_rng(81009123);resampled=[]
        for block in range(50):
            if len(stochastic):
                indices=rng.integers(0,len(stochastic),(100,len(stochastic)))
                c=(counts[indices].sum(1)+fixed).astype(float)
            else:
                c=np.repeat(fixed[None],100,axis=0).astype(float)
            tp,fp,fn=c.T
            denom=2*tp+fp+fn
            f1=np.divide(2*tp,denom,out=np.full(100,np.nan),where=denom>0)
            precision=np.divide(tp,tp+fp,out=np.full(100,np.nan),where=tp+fp>0)
            recall=np.divide(tp,tp+fn,out=np.full(100,np.nan),where=tp+fn>0)
            resampled.append(np.stack([precision,recall,f1],1))
        boot=np.concatenate(resampled)
        result['bootstrap_95ci']={k:np.nanquantile(boot[:,i],[.025,.975]).tolist() if np.isfinite(boot[:,i]).any() else None
                                  for i,k in enumerate(['precision','recall','f1'])}
        result['bootstrap_unit']='independent full input trajectory';result['bootstrap_replicates']=5000
        result['bootstrap_stochastic_trajectories']=len(stochastic)
        result['deterministic_zero_controls_held_fixed']=len(deterministic)
        result['f1_interpretable']=result['reference_events']>0
        return result
    low=[r for r in records if r['low_density']]
    groups={}
    for axis in ['family','duration_s','A','reference_spikes']:
        groups[axis]={str(value):summarize([r for r in low if r[axis]==value]) for value in sorted(set(r[axis] for r in low))}
    zero=[r for r in low if r['reference_spikes']==0]
    result=dict(finished=True,created=datetime.now().isoformat(),manifest_sha256=sha(work/'confirmatory/manifest.json'),
                checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),model_definition='Full quadratic W including trainable diagonal',
                sample_interval_ms=1.,reference_dt_ms=.1,event_tolerance_ms=5,voltage_mask_ms=10,
                test_inputs_used_for_training=False,test_inputs_used_for_checkpoint_or_reset_selection=False,
                all_generated=summarize(records),low_density=summarize(low),zero_spike_controls=summarize(zero),
                excluded_high_density_cases=len(records)-len(low),target_low_density_events=300,target_met=sum(r['reference_spikes'] for r in low)>=300,
                groups=groups,records=records)
    assert sha(checkpoint)==manifest['checkpoint_sha256']
    write(work/'evaluation.json',result)
    print('RESULT',result['low_density'],flush=True)


if __name__=='__main__':main()
