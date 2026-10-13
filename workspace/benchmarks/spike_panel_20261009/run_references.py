"""Parallel independent NEURON references; one solver thread per trajectory."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
from datetime import datetime
import numpy as np
from common_panel import config,read,write,sha

STATE=None


def initialize(root,out):
    global STATE
    root=Path(root);out=Path(out)
    sys.path.insert(0,str(root/'scripts'))
    import sample_dbnn_data_repro as sample
    from neuron import h,coreneuron
    state=sample.build_worker_network(config(root))
    coreneuron.enable=False
    pc=h.ParallelContext();pc.nthread(1);pc.thread_busywait(0);pc.set_maxstep(10)
    for ns in state['netstims']:ns.number=0
    for nc in state['netcons']:nc.weight[0]=0
    active=[]
    for syn in state['synapses']:
        nc=h.NetCon(None,syn);nc.delay=10;active.append(nc)
    STATE=dict(h=h,network=state,active=active,pc=pc,out=out)


def run(row):
    s=STATE;h=s['h'];out=s['out'];cid=row['case_id']
    path=out/f'{cid}_reference.json'
    if path.exists():
        result=read(path)
        if result.get('finished') and (out/f'{cid}_reference.npz').exists():
            assert result['input_sha256']==row['input_sha256']
            return result
    inp=np.load(row['input_file']);ids=inp['ids'];x=inp['x']
    import hashlib
    assert hashlib.sha256(x.tobytes()).hexdigest()==row['input_sha256']
    for nc in s['active']:nc.weight[0]=0
    for site in ids:s['active'][int(site)].weight[0]=1.88e-4 if site<250 else 8e-5
    vv=h.Vector().record(s['network']['cell'].soma[0](.5)._ref_v);tt=h.Vector().record(h._ref_t)
    nc=h.NetCon(s['network']['cell'].soma[0](.5)._ref_v,None,sec=s['network']['cell'].soma[0]);nc.threshold=-20
    events=h.Vector();nc.record(events)
    h.dt=.1;h.steps_per_ms=10;h.secondorder=0
    h.finitialize(-75);h.fcurrent();h.frecord_init();h.continuerun(500)
    start=time.perf_counter()
    for j,site in enumerate(ids):
        for t in np.flatnonzero(x[j]):s['active'][int(site)].event(float(500+t))
    schedule=time.perf_counter()-start
    h.continuerun(500+row['duration_ms'])
    compute=time.perf_counter()-start-schedule
    times=np.array(tt.as_numpy(),copy=True)-500;vals=np.array(vv.as_numpy(),copy=True)
    target=np.arange(row['duration_ms']+1,dtype=np.float64)
    if times[-1]<target[-1]-1e-6:
        times=np.append(times,target[-1]);vals=np.append(vals,float(s['network']['cell'].soma[0](.5).v))
    voltage=np.interp(target,times,vals).astype(np.float32)
    sp=np.array(events.as_numpy(),copy=True)-500
    sp=sp[(sp>=-1e-6)&(sp<=row['duration_ms']+1e-6)]
    minimum_isi=float(np.diff(sp).min()) if len(sp)>1 else float(row['duration_ms'])
    np.savez_compressed(out/f'{cid}_reference.npz',voltage=voltage,time_ms=target,spike_times=sp,ids=ids)
    record=dict(finished=True,case_id=cid,input_sha256=row['input_sha256'],family=row['family'],duration_ms=row['duration_ms'],
                dt_ms=.1,output_interval_ms=1.,spike_count=len(sp),spike_times_ms=sp.tolist(),minimum_isi_ms=minimum_isi,
                low_density=(len(sp)<=3 and minimum_isi>=50),schedule_ms=schedule*1000,compute_ms=compute*1000,
                gate_width_ms=row.get('gate_width_ms'),solver_threads=1)
    write(path,record)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--phase',choices=['pilot','confirmatory'],required=True);p.add_argument('--workers',type=int,default=16)
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve();folder=work/a.phase
    manifest=read(folder/'manifest.json');out=folder/'references';out.mkdir(exist_ok=True)
    records=[];started=time.perf_counter()
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(root),str(out))) as pool:
        futures={pool.submit(run,r):r for r in manifest['cases']}
        for future in as_completed(futures):
            rec=future.result();records.append(rec)
            status=dict(updated=datetime.now().isoformat(),phase=a.phase,total=len(futures),completed=len(records),
                        reference_spikes=sum(r['spike_count'] for r in records),low_density_spikes=sum(r['spike_count'] for r in records if r['low_density']),
                        elapsed_s=time.perf_counter()-started,records=sorted(records,key=lambda r:r['case_id']))
            write(folder/'reference_status.json',status)
            print('REFERENCE',len(records),len(futures),rec['case_id'],'spikes',rec['spike_count'],
                  'low_density_total',status['low_density_spikes'],flush=True)
    assert len(records)==len(manifest['cases'])
    write(folder/'reference_complete.json',dict(finished=True,**status))


if __name__=='__main__':main()
