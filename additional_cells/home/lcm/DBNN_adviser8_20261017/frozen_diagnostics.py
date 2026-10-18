"""Freeze the primary checkpoint; diagnostic resampling never changes its reset."""
from pathlib import Path
import argparse
import hashlib
import sys
import time
import numpy as np
import torch
from common8 import read,write,sha,mask,metric,aggregate,match_events,event_aggregate,reset_function
from train_full_matrix import VariableDrive,fft_length

CP_SHA='737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d'


def teacher(n,events,amp,tau):
    # Same round-to-native-grid and unique-bin convention as the frozen primary trainer.
    changes=np.zeros(n,float)
    for event in {int(round(float(e))) for e in events}:
        start=event+1
        if start<n:changes[start]-=amp
    r=np.zeros(n,float);alpha=np.exp(-1/tau)
    for i in range(1,n):r[i]=alpha*r[i-1]+changes[i]
    return r


def bootstrap_calibration(sse,norm,tp,fp,fn,size,params,nboot):
    rng=np.random.default_rng(20261017)
    samples=rng.integers(0,len(size),(nboot,len(size)))
    count=np.zeros((nboot,len(size)))
    for i in range(nboot):count[i]=np.bincount(samples[i],minlength=len(size))
    T,P,N=count@tp.T,count@fp.T,count@fn.T
    den=2*T+P+N
    f1=np.divide(2*T,den,out=np.zeros_like(T),where=den>0)
    J=(count@sse.T)/(count@size)[:,None]+.02*(count@norm.T)/len(size)+.05*(1-f1)
    best=np.argmin(J,axis=1)
    unique={}
    for index in best:
        p=params[index];key=tuple(p[k] for k in ['v_th','reset_amp','tau_reset'])
        unique[key]=unique.get(key,0)+1
    chosen=[dict(v_th=k[0],reset_amp=k[1],tau_reset=k[2],count=v,fraction=v/nboot)
            for k,v in sorted(unique.items(),key=lambda kv:-kv[1])]
    return dict(replicates=nboot,unit='Complete validation trajectory; event-free trajectories retained',
                method='Counterfactual reset selection at frozen final voltage weights; no selected model parameters overwritten',
                reset_frequencies=chosen,selected_indices=best)


@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--bootstraps',type=int,default=10000)
    a=ap.parse_args();root=a.root.resolve();out=a.work/'frozen_primary';out.mkdir(exist_ok=True)
    if (out/'complete.json').exists():return
    source=root/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag'
    cp=source/'best_checkpoint.pt';assert sha(cp)==CP_SHA
    state=torch.load(cp,map_location='cpu',weights_only=False)['model_state']
    pp={k:float(state[k]) for k in ['v_th','reset_amp','tau_reset']}
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    model=VariableDrive(state,torch.device('cuda'),1.,True)
    reset=reset_function(a.work)
    sys.path.insert(0,str(root/'benchmarks/channel_ablation_20260929'))
    from train_channel_ablation import teacher_reset
    # Check the teacher convention directly, including integer and fractional events.
    for events in [np.array([0.,4.,12.25,77.]),np.array([])]:
        baseline=teacher_reset(101,events,4.,40.)
        actual=teacher(101,events,4.,40.)
        assert np.max(abs(baseline-actual))<1e-6,(baseline[:20],actual[:20])
    validation=[];drives=[];refs=[];refspikes=[];start=time.monotonic()
    valrows=[r for r in read(source/'training_manifest.json')['cases'] if r['split']=='validation']
    assert len(valrows)==32
    for row in valrows:
        f=np.load(row['input_file']);n=6001;dense=np.zeros((500,n),np.float32);ids=f['ids']
        assert hashlib.sha256(f['x'].tobytes()).hexdigest()==row['input_sha256']
        dense[ids]=f['x']*np.where(ids[:,None]<250,1,-1)
        drive=model(torch.fft.rfft(torch.from_numpy(dense).cuda()[None],n=fft_length(n)),n)[0].cpu().numpy()
        ref=np.load(row['reference_file']);v=np.interp(np.arange(n),ref['time_ms'],ref['voltage'])
        drives.append(drive);refs.append(v);refspikes.append(ref['spike_times'])
    assert sum(map(len,refspikes))==23
    saved=read(source/'candidates/refine_ema/reset_grid.json')
    params=[r['params'] for r in saved]
    assert len(params)==288
    size=np.array([mask(len(v),s).sum() for v,s in zip(refs,refspikes)])
    grid_sse=[];grid_norm=[];grid_tp=[];grid_fp=[];grid_fn=[]
    for par in params:
        metrics=[];events=[]
        for drive,v,s in zip(drives,refs,refspikes):
            pred,ev=reset(drive,par['v_th'],par['reset_amp'],par['tau_reset'])
            metrics.append(metric(v,pred,s));events.append(match_events(s,ev))
        grid_sse.append([r['sse'] for r in metrics])
        grid_norm.append([r['sse']/max(r['reference_ss'],.04*r['n']) for r in metrics])
        for dest,key in [(grid_tp,'tp'),(grid_fp,'fp'),(grid_fn,'fn')]:dest.append([r[key] for r in events])
    arrays={k:np.array(v) for k,v in dict(sse=grid_sse,norm=grid_norm,tp=grid_tp,fp=grid_fp,fn=grid_fn,size=size).items()}
    boot=bootstrap_calibration(**arrays,params=params,nboot=a.bootstraps)
    chosen_indices=boot.pop('selected_indices')
    chosen_reset=tuple(pp[k] for k in ['v_th','reset_amp','tau_reset'])
    boot['original_reset']=pp
    boot['original_reset_fraction']=sum(x['fraction'] for x in boot['reset_frequencies']
        if tuple(x[k] for k in ['v_th','reset_amp','tau_reset'])==chosen_reset)
    np.savez_compressed(out/'validation_grid_arrays.npz',**arrays,bootstrap_selected_indices=chosen_indices)
    write(out/'validation_grid_parameters.json',params)
    comparisons={};all_arrays={}
    primary=root/'benchmarks/solver_main379_20261016'
    testrows=read(primary/'manifest.json')['cases'];assert len(testrows)==379
    for split,rows in [('validation',valrows),('main379',testrows)]:
        records=[]
        for i,row in enumerate(rows):
            cid=row['case_id']
            if split=='validation':drive,ref,s=drives[i],refs[i],refspikes[i]
            else:
                f=np.load(root/'benchmarks/random_init_main379_20261012/predictions'/f'{cid}.npz')
                drive=f['drive']
                rf=np.load(primary/'outputs/reference1/dt0.0125'/f'{cid}.npz')
                ref=rf['voltage'];s=rf['spike_times_ms']
            closed,ev=reset(drive,pp['v_th'],pp['reset_amp'],pp['tau_reset'])
            recovery=teacher(len(drive),s,pp['reset_amp'],pp['tau_reset'])
            forced=drive+recovery;keep=mask(len(ref),s)
            e_voltage=(forced-ref)[keep];e_reset=(closed-forced)[keep]
            record=dict(case_id=cid,reference_events_ms=s.tolist(),predicted_events_ms=ev.tolist(),
                reference_event_reset_voltage=metric(ref,forced,s),
                autonomous_reset_voltage=metric(ref,closed,s),drive_only_voltage=metric(ref,drive,s),
                events=match_events(s,ev),reset_disagreement_sse=float(e_reset@e_reset),
                error_cross_sum=float(e_voltage@e_reset),
                decomposition_residual_sse=float(np.sum(((closed-ref)[keep]-e_voltage-e_reset)**2)))
            records.append(record)
            all_arrays[split+'_'+cid+'_teacher_voltage']=forced
            all_arrays[split+'_'+cid+'_autonomous_voltage']=closed
        comparisons[split]=dict(inputs=len(rows),reference_spikes=sum(len(r['reference_events_ms']) for r in records),
            reference_event_reset_voltage=aggregate([r['reference_event_reset_voltage'] for r in records]),
            autonomous_reset_voltage=aggregate([r['autonomous_reset_voltage'] for r in records]),
            drive_only_voltage=aggregate([r['drive_only_voltage'] for r in records]),
            events=event_aggregate([r['events'] for r in records]),records=records)
    expected=comparisons['main379'];assert expected['reference_spikes']==680
    assert [expected['events'][k] for k in ['tp','fp','fn']]==[387,379,293]
    assert abs(expected['autonomous_reset_voltage']['pooled_rmse_mv']-.4403635)<1e-6
    assert sha(cp)==CP_SHA
    np.savez_compressed(out/'reset_comparison_arrays.npz',**all_arrays)
    write(out/'complete.json',dict(finished=True,checkpoint_sha256=CP_SHA,checkpoint_unchanged=True,
        all_parameters_and_threshold_frozen=True,validation_reference_events=23,validation_trajectories=32,
        bootstrap=boot,comparisons=comparisons,elapsed_s=time.monotonic()-start,
        scope='Reference-event reset is a voltage-only oracle diagnostic, not a deployed method or a new F1 result. Bootstrap-selected parameters are not installed.'))
    print('FROZEN_DIAGNOSTICS_COMPLETE',flush=True)


if __name__=='__main__':main()
