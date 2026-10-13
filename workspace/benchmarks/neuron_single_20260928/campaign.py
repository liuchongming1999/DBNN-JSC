"""Serial controlled NEURON campaign; each trial owns only its listed CPU cores."""
import json,os,subprocess,time
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
PY=H.parents[1]/'.venv/bin/python'
records=[]

def job(name,engine='neuron',threads=1,steps=1001,active=500,repeats=3,**kw):
    out=H/'results'/f'{name}.json';out.parent.mkdir(exist_ok=True)
    cfg=dict(engine=engine,threads=threads,steps=steps,active=active,repeats=repeats,**kw)
    if not out.exists():
        cmd=['taskset','-c',','.join(map(str,range(48,48+threads))),str(PY),str(H/'neuron_worker.py'),'--output',str(out)]
        for k,v in cfg.items():
            cmd += ['--'+k.replace('_','-')]+([str(x) for x in v] if isinstance(v,list) else [str(v)])
        env={**os.environ,'OMP_NUM_THREADS':str(threads),'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        print('START',name,flush=True)
        with out.with_suffix('.log').open('w') as f:
            try:
                p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
                if not out.exists():out.write_text(json.dumps(dict(finished=False,error=f'process exit {p.returncode}; see log',config=cfg)))
            except subprocess.TimeoutExpired:out.write_text(json.dumps(dict(finished=False,error='600s timeout',config=cfg)))
    r=json.loads(out.read_text());r['name']=name;records.append(r)
    (H/'campaign_status.json').write_text(json.dumps(dict(updated=time.strftime('%FT%T%z'),complete=False,completed=len(records),latest=name,records=[{k:r.get(k) for k in ('name','finished','error')} for r in records]),indent=2))
    print('DONE',name,r.get('finished'),r.get('timing',{}).get('solver_call',{}).get('median_ms'),flush=True)
    return r

def best(rows):
    rows=[r for r in rows if r.get('finished')]
    return min(rows,key=lambda r:r['timing']['solver_call']['median_ms'])['config']

def main():
    # Screening uses 0.3 s; the held-out confirmation lasts 1 or 6 s.
    pilot=[]
    for n in [1,2,4,8]:pilot.append(job(f'pilot_neuron_n{n}',threads=n,steps=301,repeats=3))
    pilot.append(job('pilot_cache0',cache=0,steps=301,repeats=3))
    ms=[]
    for busy in [0,1]:
        for n in [2,4,8]:ms.append(job(f'pilot_ms_n{n}_busy{busy}',engine='multisplit',threads=n,busywait=busy,steps=301,repeats=3))
    core=[job(f'pilot_core_n{n}',engine='core',threads=n,steps=301,repeats=3) for n in [1,2,4,8]]
    cv=[job(f'pilot_cv_{a:g}',engine='cvode',atol=a,steps=301,repeats=3) for a in [1e-3,1e-4,1e-5]]
    job('original_mechanisms',mechanisms='original',steps=301,repeats=1)
    chosen={'multisplit':best(ms),'core':best(core)}
    # CVode tolerance is fixed a priori at 1e-4, not selected for best time.
    chosen['cvode']=dict(engine='cvode',threads=1,atol=1e-4)
    (H/'selected_neuron.json').write_text(json.dumps(chosen,indent=2))
    for T in [1001,6001]:
        for A in [32,128,500]:
            job(f'main_A{A}_T{T}_neuron',active=A,steps=T)
            for engine,cfg in chosen.items():
                options={k:v for k,v in cfg.items() if k in ['threads','busywait','cache','dt','atol']}
                job(f'main_A{A}_T{T}_{engine}',engine=engine,active=A,steps=T,**options)
    # Time-step/accuracy frontier on the same full-site 1 s input.
    for dt in [.5,.2,.05,.025,.0125]:job(f'dt_{dt:g}',dt=dt,repeats=3)
    for a in [1e-3,1e-5]:job(f'cv_{a:g}',engine='cvode',atol=a,repeats=3)
    for rate in ['quiet','low','high']:
        job(f'rate_{rate}_neuron',rate=rate)
        for engine in ['multisplit','cvode']:
            cfg=chosen[engine];options={k:v for k,v in cfg.items() if k in ['threads','busywait','cache','dt','atol']}
            job(f'rate_{rate}_{engine}',engine=engine,rate=rate,**options)
    # Three reconstructed held-out inputs, including spiking trajectories.
    job('fidelity_3x6s',steps=6001,samples=[0,1,2],repeats=1)
    job('batch_serial_8x1s',steps=1001,samples=list(range(8)),repeats=3)
    status=json.loads((H/'campaign_status.json').read_text());status['complete']=True
    status['finished']=time.strftime('%FT%T%z');(H/'campaign_status.json').write_text(json.dumps(status,indent=2))
    print('CAMPAIGN COMPLETE',flush=True)

if __name__=='__main__':main()
