"""Validated CPU backend, random-init 500ms-window checkpoints, full prepared computation."""
import argparse,hashlib,os,random,subprocess,sys,time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from common import read,write,sha,metric,aggregate,raster,sample_state
from specs import configurations
from gpu_benchmark import stats,checks

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--threads',type=int,choices=[1,8],required=True);p.add_argument('--repeats',type=int,default=30)
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve();nt=a.threads;out=work/f'cpu{nt}';out.mkdir(exist_ok=True)
    sys.path[:0]=[str(root/'benchmarks/cpu_20260929'),str(root/'benchmarks/training_refinement_20260929')]
    from cpu_backend import CPUModel
    from benchmark_cpu import methods,measure_group
    from train_refinement import match_events,event_aggregate
    cpus=list(range(72,72+nt));os.sched_setaffinity(0,cpus);torch.set_num_threads(nt);torch.set_num_interop_threads(1)
    specs={n:v for n,v in configurations(root,work).items() if n!='U0'};states={n:sample_state(torch.load(path,map_location='cpu',weights_only=False)['model_state'],dt) for n,(dt,path) in specs.items()}
    cfgs=methods();rng=np.random.default_rng(934121);cal=[]
    for name,(dt,path) in specs.items():
        for A in [64,128,256,500]:
            ids=np.sort(rng.choice(500,A,replace=False)).astype(np.int64)
            for duration in [1000,4000,16000]:
                raw=rng.poisson(.025,(A,duration+1)).astype(np.float32);raw[:,0]=0;xx=raster(raw,dt);T=xx.shape[-1];dense=np.zeros((500,T),np.float32);dense[ids]=xx*(1-2*(ids[:,None]>=250))
                m=CPUModel(states[name],ids,T);oracle=m.run(dense,nt,0,0,0).copy();validation={tag:checks(m.run(dense,nt,**cfg),oracle,m.threshold) for tag,cfg in cfgs.items()};allowed={tag:cfgs[tag] for tag,v in validation.items() if v['passed']}
                if not allowed:raise RuntimeError(('No CPU backend',nt,name,A,T))
                timing=measure_group(m,dense,nt,allowed,7);selected=min(timing,key=lambda k:timing[k]['wall']['median_ms']);cal.append(dict(config=name,dt_ms=dt,A=A,T=T,selected=selected,validation=validation,timing=timing));m.close()
                write(out/'policy.json',dict(finished=False,records=cal));print('TUNE',nt,name,A,duration,selected,flush=True)
    write(out/'policy.json',dict(finished=True,records=cal,rule='Fastest complete forward among equivalence-checked options on independent synthetic inputs, nearest log-duration for matching A and dt.'))
    runtime=dict(started=datetime.now().isoformat(),threads=nt,physical_cpus=cpus,shared=True,repeats=a.repeats,variants={n:dict(dt_ms=dt,checkpoint=str(path),checkpoint_sha256=sha(path),kernels_per_site=1) for n,(dt,path) in specs.items()},wall_scope='Prepared signed CPU raster -> active-site lookup/filter -> complete quadratic/linear/bias -> event reset -> preallocated CPU voltage. External input packing, coefficient/weight preparation and allocation excluded.',precision='FP32 AVX512/FMA and MKL SGEMM; original CPU implementation',environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES','OMP_WAIT_POLICY']},topology=subprocess.run(['lscpu','-e=CPU,CORE,SOCKET,NODE'],text=True,capture_output=True).stdout,loadavg=os.getloadavg())
    write(out/'runtime.json',runtime);src=root/'benchmarks/dt_resolution_20260929/test';rows=read(src/'manifest.json')['cases'];records=[];order=random.Random(930179)
    for row in rows:
        cid=row['case_id'];data=np.load(row['input_file']);ids=data['ids'].astype(np.int64);raw=data['x'];assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256']
        ref=np.load(src/'references'/(cid+'_reference.npz'));models={};dense={};selected={};validation={};timing={n:[] for n in specs};cpu_time={n:[] for n in specs}
        for name,(dt,path) in specs.items():
            xx=raster(raw,dt);T=xx.shape[-1];dense[name]=np.zeros((500,T),np.float32);dense[name][ids]=xx*(1-2*(ids[:,None]>=250));m=CPUModel(states[name],ids,T);models[name]=m
            oracle=np.load(work/'gpu'/name/(cid+'.npz'))['voltage'];valid={tag:checks(m.run(dense[name],nt,**cfg),oracle,m.threshold) for tag,cfg in cfgs.items()};validation[name]=valid
            near=min([c for c in cal if c['config']==name and c['A']==len(ids)],key=lambda c:abs(np.log(c['T']/T)));allowed=[tag for tag in near['timing'] if valid[tag]['passed']]
            if not allowed:raise RuntimeError(('CPU oracle mismatch',nt,name,cid,valid))
            selected[name]=min(allowed,key=lambda tag:near['timing'][tag]['wall']['median_ms'])
            for _ in range(3):m.run(dense[name],nt,**cfgs[selected[name]])
        for _ in range(a.repeats):
            names=list(specs);order.shuffle(names)
            for name in names:
                pc=time.process_time();t=time.perf_counter();models[name].run(dense[name],nt,**cfgs[selected[name]]);end=time.perf_counter();timing[name].append((end-t)*1000);cpu_time[name].append((time.process_time()-pc)*1000)
        rec=dict(case_id=cid,A=len(ids),duration_s=row['duration_ms']/1000,input_sha256=row['input_sha256'],variants={})
        for name,(dt,path) in specs.items():
            m=models[name];pv=m.output.copy();ev=np.flatnonzero(pv>=m.threshold)*dt;oracle=np.load(work/'gpu'/name/(cid+'.npz'))['voltage'];final=checks(pv,oracle,m.threshold);assert final['passed'];mm=metric(ref['voltage'][::2],pv[::round(1/dt)],ref['spike_times'],1.);ee=match_events(ref['spike_times'],ev)
            (out/name).mkdir(exist_ok=True);np.savez_compressed(out/name/(cid+'.npz'),voltage=pv,drive=m.drive(),reset_events_ms=ev);rec['variants'][name]=dict(dt_ms=dt,wall=stats(timing[name]),process_cpu=stats(cpu_time[name]),metrics=mm,events=ee,numerical=final,all_backend_checks=validation[name],backend=selected[name]);m.close()
        records.append(rec);write(out/(cid+'.json'),rec);write(out/'status.json',dict(completed=len(records),total=len(rows)));print('CASE',nt,cid,flush=True)
    summary={n:dict(metrics=aggregate([r['variants'][n]['metrics'] for r in records]),events=event_aggregate([r['variants'][n]['events'] for r in records]),wall_ms=sum(r['variants'][n]['wall']['median_ms'] for r in records)) for n in specs}
    write(out/'complete.json',dict(finished=True,records=records,summary=summary,runtime=runtime,completed=datetime.now().isoformat()));print('COMPLETE',nt,summary,flush=True)

if __name__=='__main__':main()
