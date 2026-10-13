"""Paired prepared GPU measurements and independent full-curve oracles."""
import argparse,gc,hashlib,os,random,subprocess,sys,time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from common import read,write,sha,metric,aggregate
from specs import configurations

def stats(x):return dict(median_ms=float(np.median(x)),q1_ms=float(np.quantile(x,.25)),q3_ms=float(np.quantile(x,.75)),runs_ms=x)
def checks(v,oracle,threshold):
    a=np.asarray(v,float);b=np.asarray(oracle,float);e=a-b
    r=dict(max_abs_mv=float(abs(e).max()),rmse_mv=float(np.sqrt(np.mean(e**2))),events_equal=bool(np.array_equal(a>=threshold,b>=threshold)),finite=bool(np.isfinite(a).all()))
    r['passed']=r['finite'] and r['events_equal'] and r['max_abs_mv']<.001
    return r

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--repeats',type=int,default=30);a=p.parse_args();root=a.root.resolve();work=a.work.resolve();out=work/'gpu';out.mkdir(exist_ok=True)
    sys.path[:0]=[str(root/'benchmarks/compute_fair_20260929'),str(root/'benchmarks/sparse_20260928'),str(root/'benchmarks/channel_ablation_20260929'),str(root/'benchmarks/training_refinement_20260929')]
    from backends import subset_state
    from train_channel_ablation import reset
    from train_refinement import match_events,event_aggregate
    from gpu_backend import ResolutionModel,Graph,options
    os.sched_setaffinity(0,{72});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    specs=configurations(root,work);states={n:torch.load(path,map_location='cuda',weights_only=False)['model_state'] for n,(_,path) in specs.items()}
    locks={n:read(path.parent/'final_selection.json') for n,(_,path) in specs.items()};assert all(locks[n]['finished'] and locks[n]['checkpoint_sha256']==sha(path) and locks[n]['protocol']['loss_burn_in_ms']==150 and locks[n]['protocol']['training_window_ms']==500 for n,(_,path) in specs.items());b7=locks['B7']
    shapes={k:tuple(v.shape) for k,v in states['B2'].items()};assert all({k:tuple(v.shape) for k,v in st.items()}==shapes for st in states.values())
    def oracle(st,x,dt):
        T=(x.shape[-1]-1)*round(1/dt)+1;nfft=1<<(2*T-2).bit_length();t=torch.arange(T,device='cuda')*dt
        k=st['omega'][:,None]*(1-torch.exp(-t/st['tau_rise'][:,None]))*torch.exp(-t/st['tau_decay'][:,None])
        expanded=torch.zeros((1,x.shape[1],T),device='cuda');expanded[...,::round(1/dt)]=x
        y=torch.fft.irfft(torch.fft.rfft(expanded,n=nfft)*torch.fft.rfft(k,n=nfft),n=nfft)[...,:T].transpose(1,2).contiguous()
        u=((y@st['bilinear.weight'][0])*y).sum(-1)+y.sum(-1)+st['bilinear.bias'];v,ev=reset(u[0].cpu().numpy(),float(st['v_th']),float(st['reset_amp']),float(st['tau_reset'])/dt)
        return v,u[0].cpu().numpy(),ev*dt
    rng=np.random.default_rng(934121);cal=[]
    for name in ['B2','B6','B7']:
        dt=specs[name][0]
        for A in [64,128,256,500]:
            ids=np.sort(rng.choice(500,A,replace=False));ii=torch.as_tensor(ids,device='cuda');st=subset_state(states[name],ii)
            for duration in [1000,4000,16000]:
                raw=rng.poisson(.025,(1,A,duration+1)).astype(np.int8);raw[...,0]=0
                x=torch.from_numpy(raw*(1-2*(ids[None,:,None]>=250)).astype(np.int8)).contiguous().cuda();ov,ou,oe=oracle(st,x,dt);timing={};validation={}
                candidates=options(dt);random.Random(5171+duration+A).shuffle(candidates)
                for mode,block in candidates:
                    tag=f'{mode}_{block}';m=ResolutionModel(st,x,dt,mode,block);v=m();torch.cuda.synchronize();valid=checks(v[0].cpu().numpy(),ov,m.th);validation[tag]=valid
                    if valid['passed']:
                        g=Graph(m,events=True);tt=[]
                        for _ in range(7):g();torch.cuda.synchronize();tt.append(g.elapsed_ms())
                        timing[tag]=stats(tt);del g
                    del m,v
                if not timing:raise RuntimeError(('No valid backend',name,A,duration))
                selected=min(timing,key=lambda n:timing[n]['median_ms']);cal.append(dict(config=name,dt_ms=dt,A=A,input_steps=duration+1,selected=selected,timing=timing,validation=validation))
                write(work/'gpu_policy.json',dict(records=cal,finished=False));print('TUNE',name,A,duration,selected,flush=True);del x,ov,ou,st;gc.collect();torch.cuda.empty_cache();st=subset_state(states[name],ii)
    write(work/'gpu_policy.json',dict(finished=True,records=cal,rule='Fastest complete CUDA-device forward among independently FFT/reset-validated scan implementations on synthetic inputs, nearest log-duration for matching A and dt. No main31 accuracy used.'))
    runtime=dict(started=datetime.now().isoformat(),variants={n:dict(dt_ms=dt,checkpoint=str(path),checkpoint_sha256=sha(path),kernels_per_site=1) for n,(dt,path) in specs.items()},gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),device=torch.cuda.get_device_name(),shared=True,cpu_affinity=[72],repeats=a.repeats,wall_scope='Prepared complete CUDA Graph replay plus completion synchronization. All filter, quadratic/linear/bias and reset computation included. CPU packing, upload and download excluded.',device_scope='CUDA events inside the complete graph.',resource_start=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'],capture_output=True,text=True).stdout,checkpoint_locked_at={n:l['locked_at'] for n,l in locks.items()},training_window_ms=500,loss_burn_ms=150,precision='FP32 IEEE',scripts={p.name:sha(p) for p in work.glob('*.py')})
    write(out/'runtime.json',runtime);src=root/'benchmarks/dt_resolution_20260929/test';rows=read(src/'manifest.json')['cases'];records=[];order=random.Random(930179)
    for row in rows:
        cid=row['case_id'];data=np.load(row['input_file']);raw=data['x'];ids=data['ids'].astype(np.int64)
        assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256'];assert hashlib.sha256(ids.astype(np.int32).tobytes()).hexdigest()==row['ids_sha256']
        x=torch.from_numpy(raw.astype(np.int8)*(1-2*(ids[:,None]>=250)).astype(np.int8))[None].contiguous().cuda();ii=torch.as_tensor(ids,device='cuda');ref=np.load(src/'references'/(cid+'_reference.npz'));holders={};preds={};numerical={};mm={};ee={};selected={}
        for name,(dt,path) in specs.items():
            candidates=[c for c in cal if c['dt_ms']==dt and c['A']==len(ids)];near=min(candidates,key=lambda c:abs(np.log(c['input_steps']/raw.shape[1])))
            st=subset_state(states[name],ii);ov,ou,oe=oracle(st,x,dt);attempts={}
            # Keep the timing order fixed by synthetic calibration. A fallback
            # only enforces implementation equivalence to this checkpoint.
            for tag in sorted(near['timing'],key=lambda k:near['timing'][k]['median_ms']):
                mode,block=tag.rsplit('_',1);m=ResolutionModel(st,x,dt,mode,int(block));v=m();torch.cuda.synchronize();pv=v[0].cpu().numpy().copy();u=m.drive[0].cpu().numpy().copy();valid=checks(pv,ov,m.th);valid['drive_max_abs_mv']=float(np.max(abs(u-ou)));attempts[tag]=valid.copy()
                if valid['passed']:selected[name]=tag;break
                del m,v
            else:raise RuntimeError((cid,name,attempts))
            valid['policy_first_choice']=near['selected'];valid['attempts']=attempts
            ev=np.flatnonzero(pv>=m.th)*dt;numerical[name]=valid;preds[name]=dict(voltage=pv,drive=u,reset_events_ms=ev)
            mm[name]=metric(ref['voltage'][::2],pv[::round(1/dt)],ref['spike_times'],1.);ee[name]=match_events(ref['spike_times'],ev);holders[name]=dict(model=m,plain=Graph(m),events=Graph(m,events=True))
        wall={n:[] for n in specs};dev={n:[] for n in specs}
        for _ in range(a.repeats):
            names=list(specs);order.shuffle(names)
            for name in names:
                h=holders[name];torch.cuda.synchronize();t=time.perf_counter();h['plain']();torch.cuda.synchronize();wall[name].append((time.perf_counter()-t)*1000)
                h['events']();torch.cuda.synchronize();dev[name].append(h['events'].elapsed_ms())
        rec=dict(case_id=cid,A=len(ids),duration_s=row['duration_ms']/1000,input_sha256=row['input_sha256'],variants={})
        for name,(dt,path) in specs.items():
            timed=holders[name]['model'].out[0].cpu().numpy();assert checks(timed,preds[name]['voltage'],holders[name]['model'].th)['passed']
            (out/name).mkdir(exist_ok=True);np.savez_compressed(out/name/(cid+'.npz'),**preds[name]);rec['variants'][name]=dict(dt_ms=dt,wall=stats(wall[name]),device=stats(dev[name]),metrics=mm[name],events=ee[name],numerical=numerical[name],backend=selected[name])
        records.append(rec);write(out/(cid+'.json'),rec);write(out/'status.json',dict(completed=len(records),total=len(rows)));print('CASE',cid,flush=True)
        del holders,h,m,x,v;gc.collect();torch.cuda.empty_cache()
    summary={n:dict(metrics=aggregate([r['variants'][n]['metrics'] for r in records]),events=event_aggregate([r['variants'][n]['events'] for r in records]),wall_ms=sum(r['variants'][n]['wall']['median_ms'] for r in records),device_ms=sum(r['variants'][n]['device']['median_ms'] for r in records)) for n in specs}
    write(out/'complete.json',dict(finished=True,records=records,summary=summary,runtime=runtime,completed=datetime.now().isoformat()));print('COMPLETE',summary,flush=True)

if __name__=='__main__':main()
