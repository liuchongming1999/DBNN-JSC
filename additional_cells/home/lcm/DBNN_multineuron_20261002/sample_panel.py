"""Dense 500-site independent panels and reference-selected 31-window test."""
import argparse,hashlib,json,multiprocessing as mp,os,time,fcntl
from pathlib import Path
import numpy as np
from models import REGISTRY,build,run
from screen import write
S={}

def initialize(work,model,profile,out):
    available=list(range(0,48))+list(range(96,128))
    os.sched_setaffinity(0,{available[os.getpid()%len(available)]})
    work=Path(work);env=build(work,model,profile)
    condition=json.loads((Path(out)/'reference_condition.json').read_text());op=condition['operating_point']
    S.update(env=env,op=op,out=Path(out),condition_hash=condition['condition_sha256'])

def sample(row):
    out=S['out'];op=S['op'];npz=out/'references'/(row['case_id']+'_reference.npz');recp=npz.with_suffix('.json')
    if recp.exists() and npz.exists():return json.loads(recp.read_text())
    d=np.load(row['input_file']);ids=d['ids'];x=d['x']
    assert hashlib.sha256(x.tobytes()).hexdigest()==row['input_sha256']
    rr=run(S['env'],row['duration_ms'],ids,x,row['input_interval_ms'],op['weight_us'],op['inhibition_ratio'],op['holding_current_na'])
    target=np.arange(round(row['duration_ms']/.25)+1)*.25;v=np.interp(target,rr['time_ms'],rr['voltage']).astype(np.float32)
    np.savez_compressed(npz,voltage=v,time_ms=target,spike_times=rr['spike_times'],ids=ids)
    sp=rr['spike_times'];rec=dict(finished=True,case_id=row['case_id'],spike_count=len(sp),spike_times_ms=sp.tolist(),burst=bool(len(sp)>1 and np.min(np.diff(sp))<20),
        reference_dt_ms=.025,output_interval_ms=.25,solver_ms=rr['solver_ms'],schedule_ms=rr['schedule_ms'],input_sha256=row['input_sha256'],reference_condition_sha256=S['condition_hash'])
    write(recp,rec);return rec

def sites(A,mode,seed,positions):
    rng=np.random.default_rng(seed)
    if A==500:return np.arange(500,dtype=np.int32)
    n=A//2;exc=np.arange(250);inh=np.arange(250,500)
    # Length-stratified site index already follows original section ordering.
    if mode=='random':e=rng.choice(exc,n,False);i=rng.choice(inh,n,False)
    elif mode=='stratified':e=np.linspace(0,249,n).round().astype(int);i=e+250
    else:
        # Physical path-distance ordering is stored, not section-name ordering.
        dist=np.asarray([p['distance_um'] for p in positions]); ee=np.argsort(dist[:250]);ii=np.argsort(dist[250:])+250
        if mode=='distal':ee=ee[::-1];ii=ii[::-1]
        e=ee[:n];i=ii[:n]
    return np.sort(np.r_[e,i]).astype(np.int32)

def make_row(out,cid,duration,A,hz,mode,seed,dt,split,positions,pulses=0,gain=1):
    ids=sites(A,mode,seed,positions);rng=np.random.default_rng(seed+37);T=round(duration/dt)+1
    lam=np.full((A,T),hz*dt/1000.)
    # Finite low-rate transients over ongoing Poisson background, not a
    # stopped-input tail. Per-case event multiplicity represents fixed weight.
    if pulses:
        centers=np.linspace(duration*.2,duration*.8,pulses)
        for c in centers:
            start=round(c/dt);end=round((c+40)/dt);lam[ids<250,start:end]+=min(hz*6,120)*dt/1000.
    x=rng.poisson(lam).astype(np.uint8)*gain;x[:,0]=0;x[:,-1]=0
    p=out/'inputs'/(cid+'.npz');np.savez_compressed(p,x=x,ids=ids)
    return dict(case_id=cid,input_file=str(p),reference_file=str(out/'references'/(cid+'_reference.npz')),reference_output_interval_ms=.25,
        duration_ms=duration,A=A,rate_hz=hz,site_mode=mode,seed=seed,input_interval_ms=dt,split=split,pulses=pulses,event_multiplicity=gain,
        input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),input_events=int(x.sum()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--data',type=Path,required=True);ap.add_argument('--model',choices=REGISTRY,required=True)
    ap.add_argument('--profile',default='ampa');ap.add_argument('--kind',choices=['train','test'],required=True);ap.add_argument('--dt',type=float,default=1.);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args()
    work=a.work.resolve();out=a.data.resolve()/a.model/a.profile/('train_'+str(a.dt)+'ms' if a.kind=='train' else 'test');out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'sampling.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
    op=json.loads((work/'results'/a.model/'screen/operating_point.json').read_text())
    condition=dict(model=a.model,modeldb=REGISTRY[a.model]['modeldb'],profile=a.profile,operating_point=op,reference_dt_ms=.025,warmup_ms=2000.)
    condition['condition_sha256']=hashlib.sha256(json.dumps(condition,sort_keys=True).encode()).hexdigest()
    if (out/'reference_condition.json').exists():
        assert json.loads((out/'reference_condition.json').read_text())['condition_sha256']==condition['condition_sha256'],'A new physical condition must use a fresh dataset directory'
    else:write(out/'reference_condition.json',condition)
    if (out/'complete.json').exists():return
    for sub in ['inputs','references']:(out/sub).mkdir(exist_ok=True)
    cov=json.loads((work/'results'/a.model/'screen/sites.json').read_text());positions=cov['sites'];rows=[]
    if (out/'candidate_manifest.json').exists():rows=json.loads((out/'candidate_manifest.json').read_text())['cases']
    else:
        if a.kind=='train':
            for A in [64,128,256,500]:
                for hz in [10,25,50]:
                    for rep in range(8 if a.dt==1. else 4):
                        seed=610020000+list(REGISTRY).index(a.model)*100000+A*100+hz*10+rep+round(a.dt*10000)
                        mode=['random','stratified','proximal','distal'][rep%4];cid=f'train_dt{a.dt}_A{A}_hz{hz}_r{rep}'
                        nrep=8 if a.dt==1. else 4;split='validation' if rep>=nrep-2 else 'fit'
                        rows.append(make_row(out,cid,6000,A,hz,mode,seed,a.dt,split,positions,pulses=rep%4,gain=1+(rep%3)))
        else:
            for T in [1000,2000,4000,8000,16000]:
                for A in [64,128,256,500]:
                    for rep in range(9):
                        hz=[10,25,50][rep%3];mode=['random','stratified','proximal','distal'][rep%4]
                        cid=f'test_T{T}_A{A}_r{rep}';seed=810020000+list(REGISTRY).index(a.model)*100000+T*10+A+rep
                        rows.append(make_row(out,cid,T,A,hz,mode,seed,1.,'test_candidate',positions,pulses=rep%4,gain=1+rep//3))
        write(out/'candidate_manifest.json',dict(cases=rows,dataset_description='Independent neuron-specific fixed 500-site native-time Poisson panel, reference dt=.025ms; original active channels and constant holding operating point. No external learning baseline.'))
    records=[];write(out/'runtime.json',dict(workers=a.workers,cpu_pool=list(range(0,48))+list(range(96,128)),phase='reference_sampling',reference_dt_ms=.025,shared=True))
    with mp.get_context('spawn').Pool(a.workers,initializer=initialize,initargs=(str(work),a.model,a.profile,str(out))) as pool:
        for r in pool.imap_unordered(sample,rows):
            records.append(r);write(out/'status.json',dict(completed=len(records),total=len(rows),latest=r));print('SAMPLED',a.model,a.profile,a.kind,len(records),len(rows),r['case_id'],r['spike_count'],flush=True)
    if a.kind=='test':
        by={r['case_id']:r for r in records};eligible=[r for r in rows if by[r['case_id']]['spike_count']<=3 and not by[r['case_id']]['burst']]
        selected=[]
        # Lock coverage and 0/1/2/3 spike composition solely from NEURON before
        # surrogate fitting/evaluation; never rank test cases by DBNN quality.
        for count in [0,1,2,3]:
            rr=[r for r in eligible if by[r['case_id']]['spike_count']==count]
            selected+=rr[:min(4 if count else 3,len(rr))]
        for duration in [1000,2000,4000,8000,16000]:
            rr=[r for r in eligible if r['duration_ms']==duration and r not in selected];selected+=rr[:2]
        for A in [64,128,256,500]:
            if not any(r['A']==A for r in selected):selected+=next(([r] for r in eligible if r['A']==A and r not in selected),[])
        for hz in [10,25,50]:
            if not any(r['rate_hz']==hz for r in selected):selected+=next(([r] for r in eligible if r['rate_hz']==hz and r not in selected),[])
        selected=(selected+[r for r in eligible if r not in selected])[:31]
        if len(selected)<31:raise RuntimeError(('Insufficient nonburst low-spike tests',a.model,len(selected)))
        rows=selected
        for r in rows:r['split']='test';r['reference_spike_count']=by[r['case_id']]['spike_count']
        write(out/'selection.json',dict(locked_before_surrogate_evaluation=True,candidate_cases=len(records),eligible=len(eligible),selected=[r['case_id'] for r in rows],spike_histogram={str(k):sum(r['reference_spike_count']==k for r in rows) for k in range(4)},excluded='Burst ISI<20ms or >3 spikes; remaining cases selected by declared output-count/duration/A/rate coverage.'))
    write(out/'manifest.json',dict(cases=rows,dataset_description='Neuron-specific independent fixed 500-site panel; locked fit/validation/test seeds and spike-mask protocol; no channel deletion.'))
    write(out/'complete.json',dict(finished=True,cases=len(rows),records=records));print('PANEL_COMPLETE',a.model,a.profile,a.kind,len(rows),flush=True)

if __name__=='__main__':main()
