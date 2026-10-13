"""Locked candidates on fixed main31, paired prepared-compute GPU timers."""
import argparse,gc,hashlib,os,random,subprocess,sys,time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
from common import read,write,sha,metric,aggregate

def stats(x):return dict(median_ms=float(np.median(x)),q1_ms=float(np.quantile(x,.25)),q3_ms=float(np.quantile(x,.75)),runs_ms=x)

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--repeats',type=int,default=30)
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve();out=work/'gpu_single_kernel';out.mkdir(exist_ok=True)
    sys.path[:0]=[str(root/'benchmarks/compute_fair_20260929'),str(root/'benchmarks/sparse_20260928'),str(root/'benchmarks/channel_ablation_20260929'),str(root/'benchmarks/training_refinement_20260929')]
    from backend import Model,Graph
    from scan_variants import CompactScan
    from backends import subset_state
    from train_channel_ablation import reset
    from train_refinement import match_events,event_aggregate
    from prepared_backend import MultiKernelModel
    specs={
        'original_1ms':(1.,root/'benchmarks/channel_best_pipeline_20260929/final_main/best_checkpoint.pt'),
        'previous_0p5ms':(.5,root/'benchmarks/dt_resolution_20260929/fit_dt0.5/best_checkpoint.pt'),
        'balanced_1ms':(1.,work/'balanced_1ms/best_checkpoint.pt'),
        'fresh_warm_0p5ms':(.5,work/'fresh_warm_0p5ms/best_checkpoint.pt'),
        'fresh_scratch_0p5ms':(.5,work/'fresh_scratch_0p5ms/best_checkpoint.pt'),
        'b2_warm_0p5ms':(.5,work/'b2_warm_0p5ms/best_checkpoint.pt')}
    for name in list(specs)[2:]:
        selection=read(work/name/'final_selection.json');assert selection['finished'] and selection['checkpoint_sha256']==sha(specs[name][1])
    selections={name:read(work/name/'final_selection.json') for name in list(specs)[2:]}
    primary=min(['fresh_warm_0p5ms','fresh_scratch_0p5ms','b2_warm_0p5ms'],key=lambda n:selections[n]['selected']['validation']['score'])
    write(out/'primary_selection.json',dict(name=primary,locked_at=datetime.now().isoformat(),
        rule='Minimum validation selection score among B3, B4 and B6, all sharing exactly 189 fit and 56 validation cases and the original single-kernel architecture. Locked before evaluating B6 on main31; main31 results from earlier candidates were already known.',
        checkpoint_sha256=sha(specs[primary][1]),validation=selections[primary]['selected']['validation']))
    os.sched_setaffinity(0,{72});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    states={n:torch.load(path,map_location='cuda',weights_only=False)['model_state'] for n,(_,path) in specs.items()}
    assert all(not any(k.startswith('aux_') for k in st) for st in states.values())
    policy=read(root/'benchmarks/compute_fair_20260929/deployment_policy.json')
    src=root/'benchmarks/dt_resolution_20260929/test';rows=read(src/'manifest.json')['cases']
    runtime=dict(started=datetime.now().isoformat(),primary_config=primary,variants={n:dict(dt_ms=dt,path=str(p),checkpoint_sha256=sha(p),kernels_per_site=2 if 'aux_omega' in states[n] else 1) for n,(dt,p) in specs.items()},
        resource_start=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'],capture_output=True,text=True).stdout,
        gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),device=torch.cuda.get_device_name(),repeats=a.repeats,shared=True,cpu_affinity=[72],
        wall_scope='One complete prepared CUDA Graph replay plus synchronize. Input/weights/workspace ready; no CPU pack, H2D or D2H.',
        device_scope='External CUDA events captured around full filter + quadratic/linear/bias + hard event reset.',
        selection='B6 is a sequential follow-up to the earlier B0-B4 results. Its checkpoint and reset use fit/validation only, locked before its main31 evaluation. Same 31 event inputs and common 1ms observation grid; +/-10ms reference-spike mask.',
        policy='Original one-kernel-per-site architecture for every candidate. Existing synthetic-shape scan policy, complete quadratic/linear/bias fusion and event reset.',
        scripts={p.name:sha(p) for p in work.glob('*.py')})
    write(out/'runtime.json',runtime);records=[];rng=random.Random(930179)
    for row in rows:
        cid=row['case_id'];raw=np.load(row['input_file']);ids=raw['ids'].astype(np.int64);xx=raw['x']
        assert hashlib.sha256(xx.tobytes()).hexdigest()==row['input_sha256']
        assert hashlib.sha256(ids.astype(np.int32).tobytes()).hexdigest()==row['ids_sha256']
        x=torch.from_numpy(xx.astype(np.int8)*(1-2*(ids[:,None]>=250)).astype(np.int8))[None].contiguous().cuda();ii=torch.from_numpy(ids).cuda()
        holders={};preds={};numerical={};chosen={};metrics={};native_metrics={};events={}
        ref=np.load(src/'references'/(cid+'_reference.npz'))
        for name,(dt,cp) in specs.items():
            near=min([q for q in policy['records'] if q['A']==len(ids) and q['dt_ms']==dt],key=lambda q:abs(np.log(q['input_steps']/x.shape[-1])))
            mode,block=near['selected'].rsplit('_',1);chosen[name]=near['selected']
            st=subset_state(states[name],ii)
            for key in ['aux_tau_rise','aux_tau_decay','aux_omega']:
                if key in st:st[key]=st[key][ii].contiguous()
            m=MultiKernelModel(st,x,dt,mode,int(block));v=m();torch.cuda.synchronize();pred=v[0].cpu().numpy().copy();drive=m.drive[0].cpu().numpy().copy()
            T=m.T;nfft=1<<(2*T-2).bit_length();tt=torch.arange(T,device='cuda')*dt
            k=st['omega'][:,None]*(1-torch.exp(-tt/st['tau_rise'][:,None]))*torch.exp(-tt/st['tau_decay'][:,None])
            if 'aux_omega' in st:k+=st['aux_omega'][:,None]*(1-torch.exp(-tt/st['aux_tau_rise'][:,None]))*torch.exp(-tt/st['aux_tau_decay'][:,None])
            expanded=torch.zeros((1,len(ids),T),device='cuda');expanded[...,::round(1/dt)]=x
            yy=torch.fft.irfft(torch.fft.rfft(expanded,n=nfft)*torch.fft.rfft(k,n=nfft),n=nfft)[...,:T].transpose(1,2).contiguous()
            ud=((yy@st['bilinear.weight'][0])*yy).sum(-1)+yy.sum(-1)+st['bilinear.bias']
            oracle,oe=reset(ud[0].cpu().numpy(),m.th,m.amp,float(st['tau_reset'])/dt)
            er=pred-oracle;ev=np.flatnonzero(pred>=m.th)*dt
            test=dict(max_abs_mv=float(abs(er).max()),rmse_mv=float(np.sqrt(np.mean(er**2))),drive_max_abs_mv=float(np.max(abs(drive-ud[0].cpu().numpy()))),events_equal=bool(np.array_equal(ev,oe*dt)),finite=bool(np.isfinite(pred).all()))
            test['passed']=test['finite'] and test['events_equal'] and test['max_abs_mv']<.001
            if not test['passed']:raise RuntimeError((cid,name,test))
            numerical[name]=test;preds[name]=dict(voltage=pred,drive=drive,reset_events_ms=ev)
            metrics[name]=metric(ref['voltage'][::2],pred[::round(1/dt)],ref['spike_times'],1.);events[name]=match_events(ref['spike_times'],ev)
            fine_pred=pred if dt==.5 else np.interp(np.arange(len(ref['voltage']))*.5,np.arange(len(pred)),pred)
            native_metrics[name]=metric(ref['voltage'],fine_pred,ref['spike_times'],.5)
            holders[name]=dict(model=m,plain=Graph(m),events=Graph(m,events=True))
            del yy,ud,k,expanded,st
        wall={n:[] for n in specs};dev={n:[] for n in specs}
        for _ in range(a.repeats):
            names=list(specs);rng.shuffle(names)
            for name in names:
                h=holders[name];torch.cuda.synchronize();t=time.perf_counter();h['plain']();torch.cuda.synchronize();wall[name].append((time.perf_counter()-t)*1000)
                h['events']();torch.cuda.synchronize();dev[name].append(h['events'].elapsed_ms())
        rec=dict(case_id=cid,A=len(ids),duration_s=row['duration_ms']/1000,input_sha256=row['input_sha256'],variants={})
        for name in specs:
            timed=holders[name]['model'].out[0].cpu().numpy();assert np.max(abs(timed-preds[name]['voltage']))<.001
            assert np.array_equal(timed>=holders[name]['model'].th,preds[name]['voltage']>=holders[name]['model'].th)
            (out/name).mkdir(exist_ok=True);np.savez_compressed(out/name/(cid+'.npz'),**preds[name])
            rec['variants'][name]=dict(wall=stats(wall[name]),device=stats(dev[name]),numerical=numerical[name],metrics=metrics[name],fine_grid_metrics=native_metrics[name],events=events[name],backend=chosen[name],dt_ms=specs[name][0])
        records.append(rec);write(out/(cid+'.json'),rec);write(out/'status.json',dict(completed=len(records),total=31))
        print('CASE',cid,{n:round(rec['variants'][n]['metrics']['rmse_mv'],4) for n in specs},flush=True)
        del holders,h,m,x,v;gc.collect();torch.cuda.empty_cache()
    summary={n:dict(metrics=aggregate([r['variants'][n]['metrics'] for r in records]),fine_grid_metrics=aggregate([r['variants'][n]['fine_grid_metrics'] for r in records]),events=event_aggregate([r['variants'][n]['events'] for r in records]),
        wall_ms=sum(r['variants'][n]['wall']['median_ms'] for r in records),device_ms=sum(r['variants'][n]['device']['median_ms'] for r in records)) for n in specs}
    write(out/'complete.json',dict(finished=True,records=records,runtime=runtime,summary=summary,completed=datetime.now().isoformat()));print('COMPLETE',summary,flush=True)

if __name__=='__main__':main()
