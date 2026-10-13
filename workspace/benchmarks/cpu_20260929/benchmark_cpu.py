"""Synthetic timing calibration followed by locked main31 numerical validation."""
import argparse,faulthandler,hashlib,json,os,random,resource,subprocess,time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from cpu_backend import CPUModel

def read(p):return json.loads(p.read_text())
def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stat(v):
    return dict(median_ms=float(np.median(v)),mean_ms=float(np.mean(v)),q1_ms=float(np.quantile(v,.25)),q3_ms=float(np.quantile(v,.75)),min_ms=min(v),max_ms=max(v),runs_ms=v)
def methods():
    return {f'{f}_{d}':dict(filter_mode=fi,drive_mode=di,reset_mode=1)
        for f,fi in [('serial',0),('prefix16',1)] for d,di in [('mkl',0),('triangle16',1),('triangle64',2)]}
def validate(out,oracle,threshold):
    a=out.astype(float);b=oracle.astype(float);e=a-b
    events=np.flatnonzero(a>=threshold);expected=np.flatnonzero(b>=threshold)
    v=dict(max_abs_mv=float(np.max(abs(e))),rmse_mv=float(np.sqrt(np.mean(e*e))),events_equal=bool(np.array_equal(events,expected)),finite=bool(np.isfinite(a).all()),events=events.tolist(),expected_events=expected.tolist())
    v['passed']=v['finite'] and v['events_equal'] and v['max_abs_mv']<.001
    return v
def measure_group(model,dense,nt,configs,n):
    for cfg in configs.values():model.run(dense,nt,**cfg)
    wall={k:[] for k in configs};cpu={k:[] for k in configs};rng=random.Random(4206)
    for _ in range(n):
        names=list(configs);rng.shuffle(names)
        for k in names:
            pc=time.process_time();t=time.perf_counter();model.run(dense,nt,**configs[k]);end=time.perf_counter()
            wall[k].append((end-t)*1000);cpu[k].append((time.process_time()-pc)*1000)
    return {k:dict(wall=stat(wall[k]),process_cpu=stat(cpu[k])) for k in configs}

def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0));faulthandler.enable()
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--threads',type=int,required=True);ap.add_argument('--cpus',required=True);ap.add_argument('--mode',choices=['pilot','full'],default='full')
    ap.add_argument('--repeats',type=int,default=30);a=ap.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cpus=list(map(int,a.cpus.split(',')));assert len(cpus)==a.threads;os.sched_setaffinity(0,cpus)
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    checkpoint=root/'benchmarks/channel_best_pipeline_20260929/final_main/best_checkpoint.pt'
    checkpoint_sha=sha(checkpoint);assert checkpoint_sha=='888b22441881701d7544c7603c9dc2a30b1ae435f70a4de0e8e56f4083434e3f'
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False);state={k:v.detach().clone() for k,v in ck['model_state'].items()}
    manifest_path=root/'benchmarks/neuron_single_expanded_20260929/selected_20260929/manifest.json';manifest=read(manifest_path)
    configs=methods();meta=dict(started=datetime.now().isoformat(),threads=a.threads,cpus=cpus,checkpoint_sha256=checkpoint_sha,
        torch=torch.__version__,torch_config=torch.__config__.show(),environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OMP_PROC_BIND','OMP_PLACES','OMP_WAIT_POLICY','MKL_NUM_THREADS']},
        cpu_topology=subprocess.run(['lscpu','-e=CPU,CORE,SOCKET,NODE'],text=True,capture_output=True).stdout,
        cpu_model=subprocess.run(['lscpu'],text=True,capture_output=True).stdout,
        source_sha256={p.name:sha(p) for p in Path(__file__).parent.glob('*.cpp')},loadavg=os.getloadavg(),
        timer='One prepared model, B=1, CPU dense signed raster -> known active-ID lookup/filter -> full quadratic+linear+bias -> closed-loop reset -> CPU voltage. Allocation, coefficients, weight packing and compilation excluded.',
        precision='FP32, AVX512 FMA, subnormal flush-to-zero; no weight approximation',resource='Shared server; affinity selects distinct physical cores on one socket, no exclusive reservation')
    write(out/'runtime.json',meta)
    calibration=[]
    if a.mode=='full':
        # Input-independent tuning: all candidates see the same synthetic raster.
        # No main31 error or spike score participates in backend selection.
        rng=np.random.default_rng(71324)
        for K in [64,128,256,500]:
            ids=np.sort(rng.choice(500,K,replace=False)).astype(np.int64)
            for T in [1001,4001,16001]:
                dense=np.zeros((500,T),np.float32);dense[ids]=(rng.random((K,T))<.025)*(1-2*(ids[:,None]>=250))
                model=CPUModel(state,ids,T);reference=model.run(dense,a.threads,0,0,0).copy();checks={}
                for name,cfg in configs.items():
                    checks[name]=validate(model.run(dense,a.threads,**cfg),reference,model.threshold)
                    if not checks[name]['passed']:raise RuntimeError(('calibration',K,T,name,checks[name]))
                timing=measure_group(model,dense,a.threads,configs,7)
                selected=min(timing,key=lambda k:timing[k]['wall']['median_ms'])
                calibration.append(dict(A=K,T=T,selected=selected,validation=checks,timing=timing))
                write(out/'calibration.json',dict(finished=False,records=calibration))
                print('CALIBRATE',a.threads,K,T,selected,round(timing[selected]['wall']['median_ms'],4),flush=True);model.close()
        write(out/'calibration.json',dict(finished=True,records=calibration,rule='Closest log-duration among calibrated shapes with same active count; fastest complete forward on independent synthetic inputs'))
    records=[]
    cases=manifest['cases'] if a.mode=='full' else [c for c in manifest['cases'] if c['case_id'] in ['case_0020','target_018','case_0090']]
    for c in cases:
        cid=c['case_id'];p=manifest_path.parent/c['input_file']
        if not p.exists():p=manifest_path.parent/'inputs'/c['input_file']
        d=np.load(p);ids=d['ids'].astype(np.int64);raw=d['x'].astype(np.float32);K,T=raw.shape
        assert hashlib.sha256(raw.astype(np.uint8).tobytes()).hexdigest()==c['input_sha256']
        dense=np.zeros((500,T),np.float32);dense[ids]=raw*(1-2*(ids[:,None]>=250))
        gp=root/'benchmarks/channel_best_pipeline_20260929/final_main/main31'/f'dbnn_{cid}.npz'
        gj=read(gp.with_suffix('.json'));assert gj['checkpoint_sha256']==checkpoint_sha and gj['input_sha256']==c['input_sha256']
        oracle=np.load(gp)['voltage'].ravel();oracle_drive=np.load(gp)['drive'].ravel()
        start=time.perf_counter();model=CPUModel(state,ids,T);prep_ms=(time.perf_counter()-start)*1000
        checks={}
        # Every candidate is numerically checked on every input, even if not selected.
        for name,cfg in configs.items():
            v=validate(model.run(dense,a.threads,**cfg),oracle,model.threshold)
            v['drive_max_abs_mv']=float(np.max(abs(model.drive()-oracle_drive)));checks[name]=v
            if not v['passed']:raise RuntimeError((cid,name,v))
        if a.mode=='pilot':
            timing=measure_group(model,dense,a.threads,configs,3);selected=min(timing,key=lambda k:timing[k]['wall']['median_ms']);near=None
        else:
            near=min([r for r in calibration if r['A']==K],key=lambda r:abs(np.log(r['T']/T)))
            selected=near['selected'];timing=measure_group(model,dense,a.threads,{selected:configs[selected]},a.repeats)
        profile=[]
        for _ in range(10):
            times=np.zeros(3,np.float64);model.run(dense,a.threads,**configs[selected],profile=times);profile.append(times.tolist())
        pred=model.output.copy();drive=model.drive();events=np.flatnonzero(pred>=model.threshold)
        rec=dict(finished=True,case_id=cid,A=K,T=T,duration_s=c['duration_ms']/1000,input_sha256=c['input_sha256'],checkpoint_sha256=checkpoint_sha,
            threads=a.threads,selected=selected,calibration_shape=None if near is None else [K,near['T']],validation=checks,
            all_candidates_validated=True,timing=timing,selected_timing=timing[selected],preparation_ms=prep_ms,
            component_ms=dict(names=['gather_filter','drive','reset'],runs_ms=profile,median_ms=np.median(profile,axis=0).tolist()),
            reset_events_ms=events.tolist(),threshold_mv=model.threshold,input_events=c['input_events'])
        np.savez_compressed(out/f'{cid}.npz',voltage=pred,drive=drive,reset_events=events)
        write(out/f'{cid}.json',rec);records.append(rec);model.close()
        write(out/'status.json',dict(finished=False,completed=len(records),total=len(cases),latest=cid))
        print('CASE',a.threads,cid,selected,round(timing[selected]['wall']['median_ms'],4),'ms',flush=True)
    write(out/'complete.json',dict(finished=True,records=records,runtime=meta,calibration=calibration,configurations=configs))
    write(out/'status.json',dict(finished=True,completed=len(records),total=len(cases)))
    print('COMPLETE',a.threads,len(records),flush=True)

if __name__=='__main__':main()
