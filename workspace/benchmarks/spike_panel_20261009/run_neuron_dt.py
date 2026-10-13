"""Evaluate a new NEURON integration step on the frozen primary input subset.

Every input runs in a freshly constructed process. Observation grid, reference,
event detector and scoring are kept unchanged; earlier raw data are preserved.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
import multiprocessing as mp
from pathlib import Path
import sys
import numpy as np
from common_panel import read,write,sha
from score_records import grouped

DT=None


def initialize(root,out,dt):
    global DT
    from run_neuron_1ms import initialize as init
    init(root,out);DT=dt


def run(row):
    import run_neuron_1ms as old
    s=old.S
    r=s['one_case'](s['h'],s['coreneuron'],s['pc'],s['spike_nc'],s['spikes'],s['network'],s['active'],
                    row,Path('/'),s['out'],'neuron',DT,1)
    assert r['finished'] and r['input_sha256']==row['input_sha256'] and r['dt_ms']==DT
    return dict(case_id=r['case_id'],spike_count=r['spike_count'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--dt',type=float,required=True);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve();method=work/'methods'/f'neuron_dt{a.dt:g}'
    out=method/'raw';out.mkdir(parents=True,exist_ok=True)
    source=read(work/'comparison.json');manifest=read(work/'confirmatory/manifest.json')
    qualified={r['case_id'] for r in source['models']['dbnn']['records'] if r['low_density']}
    rows=[r for r in manifest['cases'] if r['case_id'] in qualified]
    assert len(rows)==379 and source['models']['dbnn']['low_density']['reference_events']==637
    protocol=dict(created=datetime.now().isoformat(),method='NEURON CPU, 1 core per input',solver_dt_ms=a.dt,output_interval_ms=1.,
                  reference_dt_ms=.1,reference_event_threshold_mv=-20.,event_tolerance_ms=5,voltage_mask_ms=10,
                  cases=len(rows),reference_events=637,selected_case_ids=[r['case_id'] for r in rows],
                  selection='Exactly the frozen primary subset selected by reference spike count <=3 and minimum ISI >=50 ms',
                  fresh_process_per_input=True,concurrent_independent_inputs=a.workers,
                  timer_scope='Accuracy campaign with concurrent cases; not a controlled serial wall-time benchmark',
                  original_comparison_sha256=sha(work/'comparison.json'),manifest_sha256=sha(work/'confirmatory/manifest.json'),
                  script_sha256=sha(Path(__file__)))
    write(method/'protocol.json',protocol);done=[]
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=initialize,
                             initargs=(str(root),str(out),a.dt),max_tasks_per_child=1) as pool:
        futures=[pool.submit(run,row) for row in rows]
        for f in as_completed(futures):
            r=f.result();done.append(r)
            write(method/'status.json',dict(updated=datetime.now().isoformat(),completed=len(done),total=len(rows),
                                            predicted_events_so_far=sum(r['spike_count'] for r in done)))
            print('SOLVED',len(done),len(rows),r['case_id'],'events',r['spike_count'],flush=True)
    sys.path[:0]=[str(root/'benchmarks/accuracy_frontier_20260929'),str(root/'benchmarks/training_refinement_20260929')]
    from common import metric,aggregate
    from train_refinement import match_events,event_aggregate
    records=[]
    for row in rows:
        cid=row['case_id'];j=read(out/f'neuron_dt{a.dt:g}_{cid}.json');pred=np.load(out/j['voltage_file'])
        ref=np.load(work/'confirmatory/references'/f'{cid}_reference.npz');rr=read(work/'confirmatory/references'/f'{cid}_reference.json')
        assert rr['low_density'] and rr['input_sha256']==row['input_sha256']==j['input_sha256']
        v=pred['voltage'];events=pred['spike_times'];assert len(v)==len(ref['voltage'])==row['duration_ms']+1
        mm=metric(ref['voltage'],v,ref['spike_times'],1.);ee=match_events(ref['spike_times'],events,tol=5)
        records.append(dict(**row,reference_spikes=rr['spike_count'],minimum_isi_ms=rr['minimum_isi_ms'],low_density=True,
                            voltage=mm,events=ee,reference_events_ms=ref['spike_times'].tolist(),predicted_events_ms=events.tolist(),
                            prediction_file=str(out/j['voltage_file'])))
    result=dict(finished=True,created=datetime.now().isoformat(),protocol=protocol,model='neuron',solver_dt_ms=a.dt,
                output_interval_ms=1.,reference_dt_ms=.1,records=records,**grouped(records,aggregate,event_aggregate))
    result['tolerance_sensitivity_ms']={str(t):event_aggregate([match_events(np.array(r['reference_events_ms']),np.array(r['predicted_events_ms']),tol=t) for r in records]) for t in [1,2,5,10,20]}
    assert result['low_density']['reference_events']==637 and sha(work/'comparison.json')==protocol['original_comparison_sha256']
    write(method/'complete.json',result)
    print('COMPLETE',result['low_density'],flush=True)


if __name__=='__main__':main()
