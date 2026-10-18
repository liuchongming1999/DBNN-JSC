"""Replay generation of the exact 149 training/validation references for cost.

No model training and no original data writes. Output checks against the frozen
training references are required before accepting the measured batch cost.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from types import SimpleNamespace
import argparse,os,sys,time,json,subprocess
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');W=R/'benchmarks/workflow_cost_20261016'
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def worker(a):
    os.sched_setaffinity(0,{64+a.shard})
    import numpy as np
    sys.path.insert(0,str(R/'benchmarks/neuron_single_expanded_20260929'))
    import run_multisplit_grid as base
    rows=json.loads((W/'offline_sources/training_manifest.json').read_text())['cases'][a.shard::a.shards]
    t=time.perf_counter();runtime=base.build(SimpleNamespace(root=R,dt=.1,threads=1,multisplit=False,verbose=0))
    h,pc,state,ncs,spike_nc,spikes,par,meta=runtime;prep=time.perf_counter()-t
    vv=h.Vector().record(state['cell'].soma[0](.5)._ref_v);tt=h.Vector().record(h._ref_t)
    for i,row in enumerate(rows):
        with np.load(row['input_file']) as f:ids=f['ids'].copy();raw=f['x'].copy()
        t=time.perf_counter()
        for nc in ncs:nc.weight[0]=0.
        for site in ids:ncs[int(site)].weight[0]=1.88e-4 if site<250 else 8e-5
        spikes.resize(0);h.dt=.1;h.steps_per_ms=10.;h.secondorder=0
        h.finitialize(-75.);h.fcurrent();h.frecord_init();h.continuerun(500.)
        for j,site in enumerate(ids):
            for n in np.flatnonzero(raw[j]):ncs[int(site)].event(float(500+n))
        pc.psolve(6500.)
        grid=np.arange(0.,6000.01,.5);voltage=np.interp(grid+500,np.asarray(tt.as_numpy()),np.asarray(vv.as_numpy())).astype(np.float32)
        ev=np.asarray(spikes.as_numpy());ev=ev[(ev>=500-1e-6)&(ev<=6500+1e-6)].copy()-500
        dest=W/'data_generation/references'/f"{row['case_id']}.npz";dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest,time_ms=grid,voltage=voltage,spike_times=ev)
        elapsed=(time.perf_counter()-t)*1000
        with np.load(row['reference_file']) as f:
            expect=np.interp(grid,f['time_ms'],f['voltage']);gap=float(np.max(np.abs(voltage-expect)))
            eq=bool(len(ev)==len(f['spike_times']) and np.allclose(ev,f['spike_times'],atol=1e-5,rtol=0))
        assert gap<.001 and eq,(row['case_id'],gap,eq)
        write(dest.with_suffix('.json'),dict(case_id=row['case_id'],split=row['split'],elapsed_ms=elapsed,
              dt_ms=.1,output_interval_ms=.5,model_prepare_s=prep,shard=a.shard,cpu=64+a.shard,
              input_sha256=row['input_sha256'],validation=dict(max_abs_mv=gap,events_equal=eq)))
        print(a.shard,i+1,len(rows),row['case_id'],flush=True)
def main(a):
    def run(s):
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OMP_WAIT_POLICY':'PASSIVE'}
        p=W/'data_generation'/f'shard{s}.log';p.parent.mkdir(exist_ok=True)
        with p.open('w') as f:
            proc=subprocess.run([sys.executable,__file__,'--worker','--shard',str(s),'--shards',str(a.shards)],stdout=f,stderr=subprocess.STDOUT,env=env)
        assert proc.returncode==0,(s,proc.returncode)
    start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.shards) as pool:
        for f in as_completed([pool.submit(run,s) for s in range(a.shards)]):f.result()
    wall=time.perf_counter()-start
    records=[json.loads(p.read_text()) for p in (W/'data_generation/references').glob('*.json')]
    assert len(records)==149 and all(x['validation']['events_equal'] for x in records)
    write(W/'data_generation/complete.json',dict(finished=True,observed_batch_elapsed_s=wall,
          physical_cores=a.shards,parallel_workers=a.shards,records=records,
          boundary='Build workers/cells, initialize and warm 500ms, schedule exact input rasters, solve at0.1ms, record and serialize0.5ms target voltages/events. Worker import/startup and oracle verification included in batch wall; unsigned raster generation excluded.'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=16);a=p.parse_args()
    if a.worker:worker(a)
    else:main(a)
