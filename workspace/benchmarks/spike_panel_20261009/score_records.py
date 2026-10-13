"""Common event aggregation; CIs resample independent input trajectories."""
import numpy as np


def summarize(records,aggregate,event_aggregate):
    if not records:return None
    s=aggregate([r['voltage'] for r in records]);s['events']=event_aggregate([r['events'] for r in records])
    s['reference_events']=sum(r['reference_spikes'] for r in records)
    s['predicted_events']=sum(len(r['predicted_events_ms']) for r in records)
    s['total_duration_s']=sum(r['duration_s'] for r in records)
    s['false_positives_per_minute']=s['events']['fp']/(s['total_duration_s']/60)
    stochastic=[r for r in records if r['family']!='zero'];det=[r for r in records if r['family']=='zero']
    counts=np.array([[r['events']['tp'],r['events']['fp'],r['events']['fn']] for r in stochastic],dtype=np.int64).reshape(-1,3)
    fixed=np.array([[r['events']['tp'],r['events']['fp'],r['events']['fn']] for r in det],dtype=np.int64).reshape(-1,3).sum(0)
    rng=np.random.default_rng(81009123);values=[]
    for _ in range(50):
        c=(counts[rng.integers(0,len(stochastic),(100,len(stochastic)))].sum(1)+fixed).astype(float) if stochastic else np.repeat(fixed[None],100,axis=0).astype(float)
        tp,fp,fn=c.T;den=2*tp+fp+fn
        cols=[np.divide(tp,tp+fp,out=np.full(100,np.nan),where=tp+fp>0),
              np.divide(tp,tp+fn,out=np.full(100,np.nan),where=tp+fn>0),
              np.divide(2*tp,den,out=np.full(100,np.nan),where=den>0)]
        values.append(np.stack(cols,1))
    values=np.concatenate(values)
    s['bootstrap_95ci']={k:np.nanquantile(values[:,i],[.025,.975]).tolist() if np.isfinite(values[:,i]).any() else None
                         for i,k in enumerate(['precision','recall','f1'])}
    s.update(bootstrap_replicates=5000,bootstrap_unit='independent full input trajectory',
             bootstrap_stochastic_trajectories=len(stochastic),deterministic_zero_controls_held_fixed=len(det),
             f1_interpretable=s['reference_events']>0)
    return s


def grouped(records,aggregate,event_aggregate):
    low=[r for r in records if r['low_density']]
    def summary(rows):return summarize(rows,aggregate,event_aggregate)
    return dict(all_generated=summary(records),low_density=summary(low),zero_spike_controls=summary([r for r in low if r['reference_spikes']==0]),
                groups={axis:{str(v):summary([r for r in low if r[axis]==v]) for v in sorted(set(r[axis] for r in low))}
                        for axis in ['family','duration_s','A','reference_spikes']})
