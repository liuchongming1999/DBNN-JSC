"""Reference-step convergence on locked inputs, including a fresh fine replay.

No surrogate is fitted and the existing reference data are left intact.
"""
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
from models import build,run,REGISTRY
from screen import write

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',choices=REGISTRY,required=True)
    p.add_argument('--profile',default='ampa')
    p.add_argument('--cpu',type=int,default=56)
    a=p.parse_args();os.sched_setaffinity(0,{a.cpu})
    sys.path[:0]=[str(a.root/'benchmarks/training_refinement_20260929')]
    from common import metric,aggregate
    from train_refinement import match_events,event_aggregate
    panel=a.data/a.model/a.profile/'test'
    rows=json.loads((panel/'manifest.json').read_text())['cases']
    selected=[]
    for n in range(4):
        eligible=[r for r in rows if r['reference_spike_count']==n]
        if eligible:selected.append(min(eligible,key=lambda r:r['duration_ms']))
    out=a.work/'results'/a.model/a.profile/'reference_step_audit'
    out.mkdir(parents=True,exist_ok=True)
    condition=json.loads((panel/'reference_condition.json').read_text())
    op=condition['operating_point'];env=build(a.work,a.model,a.profile)
    records=[]
    for row in selected:
        x=np.load(row['input_file']);ref=np.load(row['reference_file'])
        target=np.arange(round(row['duration_ms'])+1)
        reference=np.interp(target,ref['time_ms'],ref['voltage'])
        for dt in (.025,.05,.1):
            dest=out/f"{row['case_id']}_dt{dt}.json"
            if dest.exists():rec=json.loads(dest.read_text())
            else:
                rr=run(env,row['duration_ms'],x['ids'],x['x'],row['input_interval_ms'],
                       op['weight_us'],op['inhibition_ratio'],op['holding_current_na'],dt=dt)
                pred=np.interp(target,rr['time_ms'],rr['voltage'])
                rec=dict(case_id=row['case_id'],dt_ms=dt,duration_ms=row['duration_ms'],
                         reference_spikes=len(ref['spike_times']),predicted_spikes=len(rr['spike_times']),
                         metrics=metric(reference,pred,ref['spike_times'],1.),
                         events=match_events(ref['spike_times'],rr['spike_times']),
                         solver_ms=rr['solver_ms'],input_sha256=row['input_sha256'])
                write(dest,rec)
            records.append(rec)
            print(a.model,rec['case_id'],dt,rec['metrics']['rmse_mv'],rec['events'],flush=True)
    summary={str(dt):dict(metrics=aggregate([r['metrics'] for r in records if r['dt_ms']==dt]),
                         events=event_aggregate([r['events'] for r in records if r['dt_ms']==dt]),
                         wall_ms=sum(r['solver_ms'] for r in records if r['dt_ms']==dt))
             for dt in (.025,.05,.1)}
    write(out/'complete.json',dict(finished=True,collected=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
          protocol='One shortest locked test input for each available reference spike count; plain NEURON single core; common 1ms observation; saved fine-reference spike ±10ms mask and ±5ms event matching. Fresh .025ms replay checks reference identity.',
          reference_condition=condition,records=records,summary=summary))

if __name__=='__main__':main()
