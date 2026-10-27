"""Matched frozen-input solver sweep, executed only on the 5090 server.

Each worker builds one cell and processes complete independent inputs. Native
voltage/event recording is included in psolve timing; interpolation is outside.
No surrogate, reset or training configuration is changed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import threading

RESOURCE_LOCKS = {}
RESOURCE_LOCKS_GUARD = threading.Lock()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(a):
    import numpy as np
    cpus=[int(x) for x in a.cpus.split(',')]
    os.sched_setaffinity(0,cpus)
    sys.path.insert(0,str(a.root/'benchmarks/neuron_single_expanded_20260929'))
    import run_multisplit_grid as base
    a.threads=8 if a.engine=='multisplit8' else 1
    a.multisplit=False;a.verbose=0
    if a.engine=='core1':
        sys.path.insert(0,str(a.root/'benchmarks/core_threading_20260929'))
        import run_core_threads
        runtime=run_core_threads.build(a,base)
    else:
        runtime=base.build(a)
    h,pc,state,ncs,spike_nc,spikes,par,meta=runtime
    rows=read(a.work/'manifest.json')['cases'][a.shard::a.shards]
    out=a.work/'outputs'/a.engine/f'dt{a.dt:g}'
    out.mkdir(parents=True,exist_ok=True)
    for row in rows:
        path=out/f'{row["case_id"]}.json'
        if path.exists():
            old=read(path)
            assert old['input_sha256']==row['input_sha256'] and old['script_sha256'] in [sha(__file__),'f26382162fa203a26e9af3072aac3916ef287c2c785d40315cf40a213be69a3d','091a3a0a5959a7889972f788e2ae24cba591419f37a732c6b5b96043b8496873']
            continue
        with np.load(row['input_file']) as inp:
            ids=inp['ids'];raster=inp['x']
        assert hashlib.sha256(raster.tobytes()).hexdigest()==row['input_sha256']
        assert hashlib.sha256(ids.tobytes()).hexdigest()==row['ids_sha256']
        for nc in ncs:nc.weight[0]=0.
        for site in ids:ncs[int(site)].weight[0]=1.88e-4 if site<250 else 8e-5
        vv=h.Vector().record(state['cell'].soma[0](.5)._ref_v)
        tt=h.Vector().record(h._ref_t)
        target=500.+np.arange(raster.shape[1],dtype=float)
        endpoint=float(target[-1]);records=[];previous=None;previous_events=None
        repeats=1 if a.engine=='reference1' else a.repeats+1
        for rep in range(repeats):
            start=time.perf_counter();spikes.resize(0)
            h.dt=a.dt;h.steps_per_ms=1/a.dt;h.secondorder=0
            h.finitialize(-75.);h.fcurrent();h.frecord_init();h.continuerun(500.)
            warm=time.perf_counter()
            for j,site in enumerate(ids):
                for n in np.flatnonzero(raster[j]):ncs[int(site)].event(float(500+n))
            scheduled=time.perf_counter();cpu=time.process_time()
            pc.psolve(endpoint)
            solved=time.perf_counter();cpu=(time.process_time()-cpu)*1000
            times=np.asarray(tt.as_numpy());volts=np.asarray(vv.as_numpy())
            if times[-1]<endpoint-1e-5:
                times=np.append(times,endpoint);volts=np.append(volts,float(state['cell'].soma[0](.5).v))
            assert times[0]<=500 and times[-1]>=endpoint-1e-5
            voltage=np.interp(target,times,volts).astype(np.float32)
            ev=np.asarray(spikes.as_numpy());ev=ev[(ev>=500-1e-6)&(ev<=endpoint+1e-6)]-500
            assert np.isfinite(voltage).all()
            if previous is not None:
                assert np.max(np.abs(voltage-previous))<1e-4
                assert np.array_equal(ev,previous_events)
            previous=voltage.copy();previous_events=ev.copy()
            if rep or a.engine=='reference1':
                records.append(dict(solver_ms=(solved-scheduled)*1000,
                    cpu_ms=cpu,warmup_ms=(warm-start)*1000,event_schedule_ms=(scheduled-warm)*1000,
                    output_ms=(time.perf_counter()-solved)*1000,loadavg=os.getloadavg()))
        validation={}
        reference_folder=a.root/'benchmarks/spike_panel_20261009/confirmatory/references'
        if a.engine=='reference1' and a.dt==.1:
            with np.load(reference_folder/f'{row["case_id"]}_reference.npz') as frozen:
                gap=float(np.max(np.abs(voltage-frozen['voltage'])))
                events_equal=np.allclose(ev,frozen['spike_times'],atol=1e-5,rtol=0) if len(ev)==len(frozen['spike_times']) else False
            validation=dict(frozen_reference_max_voltage_gap_mv=gap,frozen_reference_events_equal=bool(events_equal))
            assert gap<1e-3 and events_equal,validation
        elif a.engine!='reference1':
            p=a.work/'outputs/reference1'/f'dt{a.dt:g}'/f'{row["case_id"]}.npz'
            with np.load(p) as f:
                gap=float(np.max(np.abs(voltage-f['voltage'])))
                equal=np.allclose(ev,f['spike_times_ms'],atol=1e-5,rtol=0) if len(ev)==len(f['spike_times_ms']) else False
            validation=dict(same_dt_single_thread_max_voltage_gap_mv=gap,same_dt_events_equal=bool(equal))
            validation['same_dt_parity_passed']=bool(gap<1e-3 and equal)
        np.savez_compressed(path.with_suffix('.npz'),voltage=voltage,time_ms=target-500,spike_times_ms=ev)
        result=dict(finished=True,engine=a.engine,dt_ms=a.dt,case_id=row['case_id'],
            input_sha256=row['input_sha256'],ids_sha256=row['ids_sha256'],
            duration_s=row['duration_s'],reference_spikes_original=row.get('reference_spikes_original',row.get('reference_spikes')),
            spike_times_ms=ev.tolist(),timing=records,median_solver_ms=float(np.median([x['solver_ms'] for x in records])),
            timing_repeats=len(records),discarded_warmup_calls=0 if a.engine=='reference1' else 1,
            validation=validation,runtime=meta,script_sha256=sha(__file__),manifest_sha256=sha(a.work/'manifest.json'))
        write(path,result)
        print(a.engine,a.dt,row['case_id'],'events',len(ev),'median_ms',result['median_solver_ms'],flush=True)


def launch_worker(a,engine,dt,shard,shards,cpus):
    log=a.work/'logs'/f'{engine}_dt{dt:g}_shard{shard}.log';log.parent.mkdir(exist_ok=True)
    command=[str(a.root/'.venv/bin/python'),str(Path(__file__).resolve()),'--worker',
        '--root',str(a.root),'--work',str(a.work),'--engine',engine,'--dt',str(dt),
        '--shard',str(shard),'--shards',str(shards),'--cpus',','.join(map(str,cpus)),
        '--repeats',str(a.repeats)]
    env={**os.environ,'OMP_NUM_THREADS':'8' if engine=='multisplit8' else '1',
        'OMP_WAIT_POLICY':'PASSIVE','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    with RESOURCE_LOCKS_GUARD:
        lock=RESOURCE_LOCKS.setdefault(tuple(cpus),threading.Lock())
    with lock, log.open('a') as f:
        p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=env)
    if p.returncode:raise RuntimeError(f'{engine} dt={dt} shard={shard} failed; {log}')
    return dict(engine=engine,dt_ms=dt,shard=shard,completed=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--worker',action='store_true');parser.add_argument('--engine',default='reference1')
    parser.add_argument('--dt',type=float,default=.1);parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=8);parser.add_argument('--cpus',default='64')
    parser.add_argument('--repeats',type=int,default=5);parser.add_argument('--phase',choices=['reference','timing'],default='reference')
    parser.add_argument('--steps',type=float,nargs='+')
    a=parser.parse_args();a.root=a.root.resolve();a.work=a.work.resolve()
    if a.worker:return worker(a)
    jobs=[]
    if a.phase=='reference':
        for dt in a.steps or [.1,.05,.025,.5,.25]:
            for shard in range(8):jobs.append(('reference1',dt,shard,8,[64+shard]))
        parallel=8
    else:
        for dt in a.steps or [.5,.25,.1,.05]:
            for shard in range(4):
                jobs.append(('multisplit8',dt,shard,4,list(range(32+8*shard,40+8*shard))))
                jobs.append(('core1',dt,shard,4,[96+shard]))
        parallel=8
    status=a.work/f'{a.phase}_status.json';done=[]
    write(status,dict(started=datetime.now().isoformat(),completed=0,total=len(jobs),finished=False))
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures=[pool.submit(launch_worker,a,*job) for job in jobs]
        for future in as_completed(futures):
            done.append(future.result())
            write(status,dict(updated=datetime.now().isoformat(),completed=len(done),total=len(jobs),finished=len(done)==len(jobs),jobs=done))
            print('COMPLETED',len(done),len(jobs),done[-1],flush=True)


if __name__=='__main__':
    main()
