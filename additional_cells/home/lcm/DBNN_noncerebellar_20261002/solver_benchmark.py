"""Single neuron, real multisplit/CoreNEURON, matched events and reference masks."""
import argparse,fcntl,json,os,time,sys,subprocess
from pathlib import Path
import numpy as np
from models import build,run,REGISTRY,register_core_sources
from screen import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--model',choices=REGISTRY,required=True);p.add_argument('--profile',default='ampa');p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--engine',choices=['neuron','multisplit','core'],required=True);p.add_argument('--dt',type=float,default=.1);p.add_argument('--repeats',type=int,default=5);p.add_argument('--smoke',action='store_true');p.add_argument('--root',type=Path,required=True)
    p.add_argument('--core-cpu',type=int,help='Dedicated physical CPU for independent single-core CoreNEURON jobs; original default remains CPU48.')
    a=p.parse_args()
    if a.core_cpu is not None:assert a.engine=='core' and 56<=a.core_cpu<=63
    work=a.work.resolve();src=a.manifest.resolve();out=work/'results'/a.model/a.profile/'solvers'/(f'{a.engine}_dt{a.dt}'+('_smoke' if a.smoke else ''));out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():return
    measurement_lock=open(out/'measurement.lock','w');fcntl.flock(measurement_lock,fcntl.LOCK_EX)
    if (out/'complete.json').exists():return
    lockname='solver_resource.lock' if a.core_cpu is None else f'solver_core_cpu{a.core_cpu}.lock'
    cpu_lock=open(work/lockname,'w');fcntl.flock(cpu_lock,fcntl.LOCK_EX)
    cpus=list(range(64,72)) if a.engine=='multisplit' else [a.core_cpu if a.core_cpu is not None else 64];os.sched_setaffinity(0,cpus)
    sys.path[:0]=[str(a.root/'benchmarks/training_refinement_20260929')]
    from common import metric,aggregate,sha
    from train_refinement import match_events,event_aggregate
    env=build(work,a.model,a.profile);h=env['h'];pc=env['pc'];pc.nthread(len(cpus));pc.thread_busywait(0)
    if a.engine=='core':register_core_sources(env)
    if a.engine=='multisplit':
        h.load_file('parcom.hoc');par=h.ParallelComputeTool();par.multisplit(1)
        env['vv'].record(env['clamp'],env['soma'](.5)._ref_v);env['tt'].record(env['clamp'],h._ref_t)
    rows=json.loads(src.read_text())['cases'];rows=rows[:2] if a.smoke else rows;op=json.loads((work/'results'/a.model/'screen/operating_point.json').read_text());records=[]
    runtime=dict(engine=a.engine,threads=len(cpus),cpus=cpus,dt_ms=a.dt,repeats=a.repeats,shared=True,loadavg=os.getloadavg(),timer='Post-warmup prepared solver call; event schedule, external input packing and output interpolation excluded; native simulator transfer inside pc.psolve retained.',manifest_sha256=sha(src),warmup_ms=2000.)
    for row in rows:
        d=np.load(row['input_file']);ref=np.load(row['reference_file']);timing=[];values=[];ev=[]
        try:
            for rep in range(a.repeats+1):
                rr=run(env,row['duration_ms'],d['ids'],d['x'],row['input_interval_ms'],op['weight_us'],op['inhibition_ratio'],op['holding_current_na'],dt=a.dt,engine=a.engine)
                if rep:timing.append(rr['solver_ms'])
            target=np.arange(round(row['duration_ms'])+1);v=np.interp(target,rr['time_ms'],rr['voltage']);r=np.interp(target,ref['time_ms'],ref['voltage'])
            if np.max(np.abs(v))>200:raise RuntimeError('Voltage exceeds 200mV: unstable solver step')
            mm=metric(r,v,ref['spike_times'],1.);events=match_events(ref['spike_times'],rr['spike_times']);np.savez_compressed(out/(row['case_id']+'.npz'),voltage=v,spike_times=rr['spike_times'])
            rec=dict(case_id=row['case_id'],metrics=mm,events=events,median_ms=float(np.median(timing)),runs_ms=timing,spike_count=len(rr['spike_times']),input_sha256=row['input_sha256']);records.append(rec)
        except Exception as exc:
            write(out/'invalid.json',dict(finished=False,case_id=row['case_id'],error=repr(exc),runtime=runtime,completed_cases=records));print('INVALID_STEP',a.model,a.engine,a.dt,repr(exc),flush=True);raise SystemExit(2)
        write(out/'progress.json',dict(completed=len(records),total=len(rows),latest=rec));print('SOLVER',a.model,a.profile,a.engine,a.dt,len(records),len(rows),rec['median_ms'],flush=True)
    write(out/'complete.json',dict(finished=True,records=records,runtime=runtime,metrics=aggregate([r['metrics'] for r in records]),events=event_aggregate([r['events'] for r in records]),wall_ms=sum(r['median_ms'] for r in records)))
    print('SOLVER_COMPLETE',a.model,a.engine,a.dt,flush=True)
    if not a.smoke:
        # This study's manifests live under /data/<owner>/<project>/<model>/<profile>/test.
        data=src.parents[3]
        publication=subprocess.run([sys.executable,str(work/'publish_evidence.py'),'--work',str(work),'--data',str(data)])
        if publication.returncode:write(out/'publication_error.json',dict(code=publication.returncode,experiment_complete=True,reason='Report postprocessing is separate from the completed solver measurement.'))

if __name__=='__main__':main()
