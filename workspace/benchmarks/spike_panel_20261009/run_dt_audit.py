"""Distinguish the numerical solver step from the output observation grid.

Each case/step starts in an independently constructed NEURON process. Output
is always sampled every 1 ms, and events use native solver threshold recording.
Original expanded-panel outputs and checkpoints are not modified.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
from pathlib import Path
import sys
import json
import numpy as np


def job(task):
    root,work,row,dt=task;root=Path(root);work=Path(work)
    cached=work/f'audit_dt{dt:g}_{row["case_id"]}.json'
    if cached.exists():
        return json.loads(cached.read_text())
    sys.path[:0]=[str(root/'benchmarks/spike_panel_20261009'),str(root/'benchmarks/neuron_single_expanded_20260929'),
                 str(root/'benchmarks/accuracy_frontier_20260929'),str(root/'benchmarks/training_refinement_20260929')]
    from run_references import initialize
    import run_references
    from run_solver_grid import one_case
    from common import metric
    from train_refinement import match_events
    initialize(str(root),str(work));state=run_references.STATE;h=state['h']
    from neuron import coreneuron
    nc=h.NetCon(state['network']['cell'].soma[0](.5)._ref_v,None,sec=state['network']['cell'].soma[0]);nc.threshold=-20;nc.delay=10
    spikes=h.Vector();nc.record(spikes)
    rec=one_case(h,coreneuron,state['pc'],nc,spikes,state['network'],state['active'],row,Path('/'),work,'neuron',dt,1)
    pred=np.load(work/rec['voltage_file']);v=pred['voltage'];events=pred['spike_times']
    reference=np.load(row['audit_reference']);t=np.arange(row['duration_ms']+1)
    ref=np.interp(t,reference['time_ms'],reference['voltage'])
    mm=metric(ref,v,reference['spike_times'],1.)
    ee=match_events(reference['spike_times'],events)
    sp=reference['spike_times']
    keep=np.ones(len(t),bool)
    for event in sp:keep[np.abs(t-event)<=10]=False
    comparison=dict(case_id=row['case_id'],family=row.get('family','legacy31'),solver_dt_ms=dt,output_interval_ms=1.,
                    native_reference_event_count=len(sp),native_prediction_event_count=len(events),
                    reference_events_ms=sp.tolist(),prediction_events_ms=events.tolist(),voltage_metrics=mm,events=ee,
                    observed_peak_mv=float(v.max()),reference_observed_peak_mv=float(ref.max()),
                    masked_voltage_bias_mv=float((v-ref)[keep].mean()),prediction_masked_mean_mv=float(v[keep].mean()),
                    reference_masked_mean_mv=float(ref[keep].mean()),native_dt_used_for_detector=True,
                    output_file=str(work/rec['voltage_file']))
    (work/f'audit_dt{dt:g}_{row["case_id"]}.json').write_text(json.dumps(comparison,indent=2)+'\n')
    return comparison


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve();work.mkdir(parents=True,exist_ok=True)
    panel=root/'benchmarks/spike_panel_20261009';rows=json.loads((panel/'confirmatory/manifest.json').read_text())['cases']
    selected=[]
    for cid in ['confirmatory_0283','confirmatory_0001','confirmatory_0515']:
        r=next(r for r in rows if r['case_id']==cid);selected.append({**r,'audit_reference':str(panel/'confirmatory/references'/f'{cid}_reference.npz')})
    old=json.loads((root/'benchmarks/dt_resolution_20260929/test/manifest.json').read_text())['cases']
    r=next(r for r in old if r['case_id']=='case_0062');selected.append({**r,'audit_reference':str(root/'benchmarks/dt_resolution_20260929/test/references/case_0062_reference.npz')})
    tasks=[(str(root),str(work),row,dt) for row in selected for dt in [.1,.25,.5,1.]]
    results=[]
    with ProcessPoolExecutor(8,mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as pool:
        for f in as_completed([pool.submit(job,t) for t in tasks]):
            r=f.result();results.append(r)
            e=r['events'];den=2*e['tp']+e['fp']+e['fn']
            e['f1']=2*e['tp']/den if den else None
            print(r['case_id'],r['solver_dt_ms'],'rmse',r['voltage_metrics']['rmse_mv'],'ref/pred',r['native_reference_event_count'],r['native_prediction_event_count'],'f1',r['events']['f1'],flush=True)
    output=dict(finished=True,reference_dt_ms=.1,output_interval_ms=1.,fresh_process_per_case_step=True,
                cases=selected,records=sorted(results,key=lambda r:(r['case_id'],r['solver_dt_ms'])))
    (work/'audit.json').write_text(json.dumps(output,indent=2)+'\n')


if __name__=='__main__':main()
