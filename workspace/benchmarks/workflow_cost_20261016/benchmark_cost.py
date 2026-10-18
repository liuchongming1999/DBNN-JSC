"""Read-only frozen-model workflow timing; runs on 5090, never trains."""
from pathlib import Path
import argparse, hashlib, json, os, sys, time, gc

ROOT=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
WORK=ROOT/'benchmarks/workflow_cost_20261016'
CP=ROOT/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt'
SHA='737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d'

def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def setup(a):
    os.sched_setaffinity(0,list(map(int,a.cpus.split(','))))
    for d in ['compute_fair_20260929','sparse_20260928','fair_window_20261005/timing_w500',
              'cpu_20260929','neuron_single_expanded_20260929','core_threading_20260929']:
        sys.path.insert(0,str(ROOT/'benchmarks'/d))
def stat(v):
    import numpy as np
    return dict(n=len(v),median_ms=float(np.median(v)),q1_ms=float(np.quantile(v,.25)),
                q3_ms=float(np.quantile(v,.75)),p95_ms=float(np.quantile(v,.95)),
                min_ms=min(v),max_ms=max(v),runs_ms=v)
def input_data(row):
    import numpy as np
    with np.load(row['input_file']) as f: ids=f['ids'].copy();raw=f['x'].copy()
    assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256']
    assert hashlib.sha256(ids.tobytes()).hexdigest()==row['ids_sha256']
    return ids,raw
def rows_for(a):
    rows=read(WORK/'manifest.json')['cases']
    if a.subset: rows=[r for r in rows if r['case_id'] in read(WORK/'protocol.json')['representative_case_ids']]
    if a.case: rows=[r for r in rows if r['case_id']==a.case]
    return rows[a.shard::a.shards]

def dbnn(a):
    import numpy as np,torch
    from gpu_backend import ResolutionModel,Graph
    from backends import subset_state
    from cpu_backend import CPUModel
    from benchmark_cpu import methods
    method=a.method;gpu=method=='gpu';nt=8 if method=='cpu8' else 1
    torch.set_num_threads(nt);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert sha(CP)==SHA
    load_start=time.perf_counter();state=torch.load(CP,map_location='cuda' if gpu else 'cpu',weights_only=False)['model_state']
    if gpu:torch.cuda.synchronize()
    load_ms=(time.perf_counter()-load_start)*1000
    locked=read(WORK/'locked_backends.json')['methods'][method]
    def prepare(ids,raw,tag):
        signed=np.ascontiguousarray(raw*np.where(ids[:,None]<250,1,-1),dtype=np.int8)
        if gpu:
            x=torch.from_numpy(signed)[None].cuda()
            st=subset_state(state,torch.as_tensor(ids.astype(np.int64),device='cuda'))
            mode,block=tag.rsplit('_',1);m=ResolutionModel(st,x,1.,mode,int(block))
            graph=Graph(m);torch.cuda.synchronize()
            return m,graph,x
        dense=np.zeros((500,raw.shape[1]),np.float32);dense[ids]=signed
        m=CPUModel(state,ids.astype(np.int64),raw.shape[1])
        return m,None,dense
    def close(obj):
        if not gpu:obj[0].close()
    def prepared(obj,tag):
        m,graph,x=obj
        if gpu:graph();torch.cuda.synchronize()
        else:m.run(x,nt,**methods()[tag])
    def resident(obj,ids,raw,tag):
        m,graph,x=obj
        signed=np.ascontiguousarray(raw*np.where(ids[:,None]<250,1,-1),dtype=np.int8)
        if gpu:
            x.copy_(torch.from_numpy(signed)[None]);graph()
            voltage=m.out[0].cpu().numpy().copy()
        else:
            x.fill(0);x[ids]=signed
            voltage=m.run(x,nt,**methods()[tag]).copy()
        events=np.flatnonzero(voltage>=m.th if gpu else voltage>=m.threshold).astype(float)
        return voltage,events
    def check(row,v,e):
        with np.load(ROOT/'benchmarks/random_init_main379_20261012/predictions'/f"{row['case_id']}.npz") as f:
            gap=float(np.max(np.abs(v-f['voltage'])))
            eq=bool(np.array_equal(e,f['predicted_events_ms']))
        assert gap<.001 and eq,(row['case_id'],gap,eq)
        return dict(max_abs_mv=gap,events_equal=eq)
    for round_index in range(a.rounds):
        rows=rows_for(a);np.random.default_rng(20261016+round_index).shuffle(rows)
        for i,row in enumerate(rows):
            cid=row['case_id'];path=WORK/'results'/method/f'round{round_index}'/f'{cid}.json'
            if path.exists():continue
            ids,raw=input_data(row);tag=locked[cid]
            t=time.perf_counter();obj=prepare(ids,raw,tag);preparation=(time.perf_counter()-t)*1000
            for _ in range(5):resident(obj,ids,raw,tag)
            times={k:[] for k in ['prepared','resident_reuse_layout','resident_new_layout']}
            for _ in range(a.repeats):
                t=time.perf_counter();prepared(obj,tag);times['prepared'].append((time.perf_counter()-t)*1000)
                t=time.perf_counter();v,e=resident(obj,ids,raw,tag);times['resident_reuse_layout'].append((time.perf_counter()-t)*1000)
            validation=check(row,v,e);close(obj);del obj
            for _ in range(a.new_repeats):
                t=time.perf_counter();obj=prepare(ids,raw,tag);v,e=resident(obj,ids,raw,tag)
                times['resident_new_layout'].append((time.perf_counter()-t)*1000)
                close(obj);del obj
            check(row,v,e)
            write(path,dict(case_id=cid,duration_s=row['duration_s'],A=row['A'],family=row['family'],
                  round_index=round_index,method=method,checkpoint_sha256=SHA,input_sha256=row['input_sha256'],
                  model_load_ms=load_ms,layout_prepare_ms=preparation,backend=tag,
                  validation=validation,timing={k:stat(v) for k,v in times.items()},loadavg=os.getloadavg()))
            if (i+1)%20==0:print(method,'round',round_index,'completed',i+1,len(rows),flush=True)
            gc.collect()
    write(WORK/'results'/method/'complete.json',dict(finished=True,rounds=a.rounds,
          inputs=len(rows_for(a)),repeats=a.repeats,new_layout_repeats=a.new_repeats,
          checkpoint_sha256=sha(CP),cpus=a.cpus,gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))

def solver(a):
    import numpy as np
    import run_multisplit_grid as base
    import types
    engine=a.method;args=types.SimpleNamespace(root=ROOT,dt=a.dt,threads=8 if engine=='neuron8' else 1,
                                             multisplit=False,verbose=0)
    t=time.perf_counter()
    if engine=='core1':
        import run_core_threads
        runtime=run_core_threads.build(args,base)
    else:runtime=base.build(args)
    h,pc,state,ncs,spike_nc,spikes,par,meta=runtime
    prep_ms=(time.perf_counter()-t)*1000
    vv=h.Vector().record(state['cell'].soma[0](.5)._ref_v);tt=h.Vector().record(h._ref_t)
    target=None
    def call(ids,raw):
        t0=time.perf_counter()
        for nc in ncs:nc.weight[0]=0.
        for site in ids:ncs[int(site)].weight[0]=1.88e-4 if site<250 else 8e-5
        spikes.resize(0);h.dt=a.dt;h.steps_per_ms=1/a.dt;h.secondorder=0
        h.finitialize(-75.);h.fcurrent();h.frecord_init();h.continuerun(500.)
        t1=time.perf_counter()
        for j,site in enumerate(ids):
            for n in np.flatnonzero(raw[j]):ncs[int(site)].event(float(500+n))
        t2=time.perf_counter();pc.psolve(float(target[-1]));t3=time.perf_counter()
        ts=np.asarray(tt.as_numpy());vs=np.asarray(vv.as_numpy())
        if ts[-1]<target[-1]-1e-5:
            ts=np.append(ts,target[-1]);vs=np.append(vs,float(state['cell'].soma[0](.5).v))
        v=np.interp(target,ts,vs).astype(np.float32)
        ev=np.asarray(spikes.as_numpy());e=ev[(ev>=500-1e-6)&(ev<=target[-1]+1e-6)].copy()-500
        t4=time.perf_counter()
        return v,e,dict(reset_and_biological_warmup_ms=(t1-t0)*1000,event_schedule_ms=(t2-t1)*1000,
                        prepared_ms=(t3-t2)*1000,output_ms=(t4-t3)*1000,resident_ms=(t4-t0)*1000)
    repset=set(read(WORK/'protocol.json')['representative_case_ids'])
    for i,row in enumerate(rows_for(a)):
        cid=row['case_id'];path=Path(a.output) if a.first_call_only else WORK/'results'/f'{engine}_dt{a.dt:g}'/f'{cid}.json'
        if path.exists():continue
        ids,raw=input_data(row);target=500.+np.arange(raw.shape[1],dtype=float)
        records=[];n=1 if a.first_call_only else 30 if cid in repset else a.repeats
        warmups=0 if a.first_call_only else 2
        for rep in range(n+warmups):
            v,e,rec=call(ids,raw)
            if rep>=warmups:records.append(rec)
        fpath=ROOT/'benchmarks/solver_main379_20261016/outputs/reference1'/f'dt{a.dt:g}'/f'{cid}.npz'
        with np.load(fpath) as f:
            gap=float(np.max(np.abs(v-f['voltage'])));eq=bool(np.array_equal(e,f['spike_times_ms']))
        assert gap<.001 and eq,(cid,gap,eq)
        write(path,dict(case_id=cid,duration_s=row['duration_s'],A=row['A'],family=row['family'],
              method=engine,dt_ms=a.dt,records=records,timing={k:stat([x[k] for x in records]) for k in records[0]},
              model_prepare_ms=prep_ms,validation=dict(max_abs_mv=gap,events_equal=eq),
              input_sha256=row['input_sha256'],cpus=a.cpus,loadavg=os.getloadavg()))
        if a.first_call_only:return
        print(engine,a.dt,a.shard,i+1,len(rows_for(a)),cid,flush=True)
    write(WORK/'results'/f'{engine}_dt{a.dt:g}'/f'complete_shard{a.shard}.json',dict(finished=True,shard=a.shard))

def cold(a):
    # A clean interpreter is launched by campaign.py; imports and runtime startup
    # are measured there. Disk input loading is outside the service boundary.
    import numpy as np,torch
    from gpu_backend import ResolutionModel,Graph
    from backends import subset_state
    from cpu_backend import CPUModel
    from benchmark_cpu import methods
    row=rows_for(a)[0];ids,raw=input_data(row);gpu=a.method=='gpu';nt=8 if a.method=='cpu8' else 1
    torch.set_num_threads(nt);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert sha(CP)==SHA;tag=read(WORK/'locked_backends.json')['methods'][a.method][row['case_id']]
    t=time.perf_counter();state=torch.load(CP,map_location='cuda' if gpu else 'cpu',weights_only=False)['model_state']
    signed=np.ascontiguousarray(raw*np.where(ids[:,None]<250,1,-1),dtype=np.int8)
    if gpu:
        x=torch.from_numpy(signed)[None].cuda();st=subset_state(state,torch.as_tensor(ids.astype(np.int64),device='cuda'))
        mode,block=tag.rsplit('_',1);m=ResolutionModel(st,x,1.,mode,int(block));graph=Graph(m);graph()
        v=m.out[0].cpu().numpy().copy();threshold=m.th
    else:
        dense=np.zeros((500,raw.shape[1]),np.float32);dense[ids]=signed
        m=CPUModel(state,ids.astype(np.int64),raw.shape[1]);v=m.run(dense,nt,**methods()[tag]).copy();threshold=m.threshold
    e=np.flatnonzero(v>=threshold).astype(float);elapsed=(time.perf_counter()-t)*1000
    with np.load(ROOT/'benchmarks/random_init_main379_20261012/predictions'/f"{row['case_id']}.npz") as f:
        assert np.max(np.abs(v-f['voltage']))<.001 and np.array_equal(e,f['predicted_events_ms'])
    write(Path(a.output),dict(case_id=row['case_id'],method=a.method,cold_runtime_ms=elapsed,
          script_pre_call_ms=(t-a.process_start)*1000,checkpoint_sha256=SHA,validation=True,
          compiler_cache='existing on-disk deployment cache; no cache deletion'))

if __name__=='__main__':
    process_start=time.perf_counter()
    p=argparse.ArgumentParser();p.add_argument('--method',required=True)
    p.add_argument('--mode',choices=['dbnn','solver','cold'],required=True);p.add_argument('--cpus',required=True)
    p.add_argument('--rounds',type=int,default=3);p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--new-repeats',type=int,default=1);p.add_argument('--dt',type=float,default=.5)
    p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--subset',action='store_true');p.add_argument('--case');p.add_argument('--output')
    p.add_argument('--first-call-only',action='store_true')
    a=p.parse_args();a.process_start=process_start;setup(a)
    if a.mode in ['dbnn','cold']:
        import torch
        with torch.inference_mode():globals()[a.mode](a)
    else:solver(a)
