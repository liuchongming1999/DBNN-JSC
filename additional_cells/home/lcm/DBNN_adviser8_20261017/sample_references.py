"""Evaluate the entire prelocked input list; no reference-output selection."""
from pathlib import Path
import argparse
import multiprocessing as mp
import os
import sys
import time
import hashlib
import fcntl
import numpy as np
from common8 import read, write, sha

S = {}


def initialize(job):
    sys.path.insert(0,job['project'])
    from models import build,run
    base=Path(job['base'])
    condition=read(base/'reference_condition.json')
    env=build(Path(job['project']),job['cell'],job['profile'],sites_file=base/'sites.json')
    rank=(mp.current_process()._identity[-1]-1)%len(job['_cpu_pool'])
    os.sched_setaffinity(0,{job['_cpu_pool'][rank]})
    S.update(env=env,run=run,condition=condition,job=job)


def sample(row):
    path=Path(row['reference_file']);meta=path.with_suffix('.json')
    if path.exists() and meta.exists():
        r=read(meta);assert r['input_sha256']==row['input_sha256'];return r
    d=np.load(row['input_file'])
    assert hashlib.sha256(d['x'].tobytes()).hexdigest()==row['input_sha256']
    op=S['condition']['operating_point'];dt=S['condition']['reference_dt_ms']
    rr=S['run'](S['env'],row['duration_ms'],d['ids'],d['x'],1.,
        op['weight_us'],op['inhibition_ratio'],op['holding_current_na'],dt=dt)
    # Keep the archived 0.25-ms output convention; the solver still uses its fixed finer step.
    t=np.arange(row['duration_ms']*4+1)*.25
    v=np.interp(t,rr['time_ms'],rr['voltage']).astype(np.float32)
    assert np.isfinite(v).all()
    np.savez_compressed(path,time_ms=t,voltage=v,spike_times=rr['spike_times'])
    r=dict(case_id=row['case_id'],input_sha256=row['input_sha256'],
           reference_sha256=sha(path),reference_dt_ms=dt,reference_output_ms=.25,
           spike_count=len(rr['spike_times']),spike_times_ms=rr['spike_times'].tolist(),
           solver_ms=rr['solver_ms'],schedule_ms=rr['schedule_ms'])
    write(meta,r);return r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--job',type=int,required=True);ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args();job=read(a.work/'locked_protocol.json')['jobs'][a.job]
    pools=[list(range(a,b)) for a,b in [(8,16),(16,24),(24,32),(48,56),(56,64),(64,72),(72,80),(88,96),(96,104),(104,112),(112,120),(120,128)]]
    job['_cpu_pool']=pools[a.job]
    base=Path(job['base']);assert sha(job['test_manifest'])==job['test_manifest_sha256']
    guard=open(base/'test/reference_sampling.lock','w')
    fcntl.flock(guard,fcntl.LOCK_EX)
    if (base/'test/complete.json').exists():return
    rows=read(job['test_manifest'])['cases'];assert len(rows)==182
    records=[];start=time.monotonic()
    with mp.get_context('spawn').Pool(a.workers,initializer=initialize,initargs=(job,)) as pool:
        for r in pool.imap_unordered(sample,rows):
            records.append(r)
            write(base/'test/status.json',dict(completed=len(records),total=len(rows),latest=r['case_id']))
            print('REFERENCE',job['cell'],job['profile'],len(records),len(rows),flush=True)
    assert {r['case_id'] for r in records}=={r['case_id'] for r in rows}
    write(base/'test/complete.json',dict(finished=True,all_inputs_retained=True,
        manifest_sha256=sha(job['test_manifest']),records=records,elapsed_s=time.monotonic()-start))


if __name__=='__main__':main()
