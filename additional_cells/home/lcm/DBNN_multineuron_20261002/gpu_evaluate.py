"""Locked per-neuron checkpoints, full optimized drive/reset, independent oracle."""
import argparse,gc,json,os,sys,time,random,subprocess,fcntl
from pathlib import Path
import numpy as np
import torch
from common import read,write,sha,metric,aggregate

def stats(x):return dict(median_ms=float(np.median(x)),runs_ms=x)
def check(a,b,th):
    e=np.asarray(a,float)-np.asarray(b,float)
    return dict(max_abs_mv=float(np.max(abs(e))),rmse_mv=float(np.sqrt(np.mean(e*e))),
                events_equal=bool(np.array_equal(np.asarray(a)>=th,np.asarray(b)>=th)),finite=bool(np.isfinite(a).all()))

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--profile',default='ampa');p.add_argument('--repeats',type=int,default=30);a=p.parse_args()
    root=a.root.resolve();work=a.work.resolve();base=work/'results'/a.model/a.profile;out=base/'gpu';out.mkdir(parents=True,exist_ok=True)
    resource_lock=open(out/'evaluation.lock','w');fcntl.flock(resource_lock,fcntl.LOCK_EX)
    if (out/'complete.json').exists():return
    sys.path[:0]=[str(root/'benchmarks/compute_fair_20260929'),str(root/'benchmarks/sparse_20260928'),str(root/'benchmarks/channel_ablation_20260929'),str(root/'benchmarks/training_refinement_20260929'),str(root/'benchmarks/resolution_frontier_20260929')]
    from backends import subset_state
    from train_channel_ablation import reset
    from train_refinement import match_events,event_aggregate
    from gpu_backend import ResolutionModel,Graph,options
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    configs={}
    for dt in [1.,.5,.25]:
        path=base/f'dt{dt}ms/best_checkpoint.pt';sel=base/f'dt{dt}ms/final_selection.json'
        if not path.exists() or not sel.exists():continue
        locked=read(sel);assert locked['finished'] and sha(path)==locked['checkpoint_sha256']
        configs[dt]=(path,torch.load(path,map_location='cuda',weights_only=False)['model_state'])
    if not configs:raise RuntimeError('No locked checkpoint')
    manifest=a.data.resolve()/a.model/a.profile/'test/manifest.json';rows=read(manifest)['cases'];assert len(rows)==31
    def oracle(st,x,dt):
        T=(x.shape[-1]-1)*round(1/dt)+1;nfft=1<<(2*T-2).bit_length();t=torch.arange(T,device=x.device)*dt
        k=st['omega'][:,None]*(1-torch.exp(-t/st['tau_rise'][:,None]))*torch.exp(-t/st['tau_decay'][:,None])
        full=torch.zeros((1,x.shape[1],T),device=x.device);full[...,::round(1/dt)]=x.float()
        y=torch.fft.irfft(torch.fft.rfft(full,n=nfft)*torch.fft.rfft(k,n=nfft),n=nfft)[...,:T].transpose(1,2).contiguous()
        u=((y@st['bilinear.weight'][0])*y).sum(-1)+y.sum(-1)+st['bilinear.bias']
        v,ev=reset(u[0].cpu().numpy(),float(st['v_th']),float(st['reset_amp']),float(st['tau_reset'])/dt)
        return v,ev*dt
    # Tune full computation on independent random events; only implementation
    # equivalence, never surrogate accuracy, decides an implementation fallback.
    rng=np.random.default_rng(61002613);policy={}
    for dt,(path,state) in configs.items():
        for A in [64,128,256,500]:
            ids=np.sort(rng.choice(500,A,False));st=subset_state(state,torch.as_tensor(ids,device='cuda'));raw=rng.poisson(.025,(1,A,4001)).astype(np.int8);raw[...,0]=0
            x=torch.from_numpy(raw*(1-2*(ids[None,:,None]>=250)).astype(np.int8)).cuda().contiguous();ov,oe=oracle(st,x,dt);candidates=[]
            for mode,block in options(dt):
                m=ResolutionModel(st,x,dt,mode,block);v=m();torch.cuda.synchronize();valid=check(v[0].cpu().numpy(),ov,m.th)
                if valid['finite'] and valid['events_equal'] and valid['max_abs_mv']<.001:
                    g=Graph(m,events=True);times=[]
                    for _ in range(7):g();torch.cuda.synchronize();times.append(g.elapsed_ms())
                    candidates.append(dict(mode=mode,block=block,device_ms=float(np.median(times)),numerical=valid));del g
                del m,v
            if not candidates:raise RuntimeError(('No valid full backend',dt,A))
            policy[f'{dt}_{A}']=sorted(candidates,key=lambda r:r['device_ms']);del st,x;gc.collect();torch.cuda.empty_cache()
    write(out/'policy.json',policy);records=[]
    for row in rows:
        d=np.load(row['input_file']);ids=d['ids'];raw=d['x'];ref=np.load(row['reference_file']);signed=raw.astype(np.int8)*(1-2*(ids[:,None]>=250)).astype(np.int8)
        assert np.max(raw)<128
        x=torch.from_numpy(signed[None]).cuda().contiguous();rec=dict(case_id=row['case_id'],duration_s=row['duration_ms']/1000,A=len(ids),input_sha256=row['input_sha256'],variants={})
        for dt,(path,state) in configs.items():
            st=subset_state(state,torch.as_tensor(ids.astype(np.int64),device='cuda'));ov,oe=oracle(st,x,dt)
            for choice in policy[f'{dt}_{len(ids)}']:
                m=ResolutionModel(st,x,dt,choice['mode'],choice['block']);v=m();torch.cuda.synchronize();pv=v[0].cpu().numpy().copy();valid=check(pv,ov,m.th)
                if valid['finite'] and valid['events_equal'] and valid['max_abs_mv']<.001:break
                del m,v
            else:raise RuntimeError(('Oracle mismatch',row['case_id'],dt))
            g=Graph(m);ge=Graph(m,events=True);wall=[];device=[]
            for _ in range(a.repeats):
                torch.cuda.synchronize();t=time.perf_counter();g();torch.cuda.synchronize();wall.append((time.perf_counter()-t)*1000)
                ge();torch.cuda.synchronize();device.append(ge.elapsed_ms())
            target=np.arange(round(row['duration_ms'])+1);rv=np.interp(target,ref['time_ms'],ref['voltage']);pred=pv[::round(1/dt)];ev=np.flatnonzero(pv>=m.th)*dt
            mm=metric(rv,pred,ref['spike_times'],1.);ee=match_events(ref['spike_times'],ev);sub=out/f'dt{dt}ms';sub.mkdir(exist_ok=True)
            np.savez_compressed(sub/(row['case_id']+'.npz'),voltage=pv,drive=m.drive[0].cpu().numpy(),reset_events_ms=ev)
            rec['variants'][str(dt)]=dict(dt_ms=dt,metrics=mm,events=ee,wall=stats(wall),device=stats(device),numerical=valid,backend=choice,checkpoint_sha256=sha(path))
            del m,v,g,ge,st;gc.collect();torch.cuda.empty_cache()
        records.append(rec);write(out/'progress.json',dict(completed=len(records),total=31,latest=rec));print('GPU_CASE',a.model,a.profile,len(records),row['case_id'],flush=True)
    summary={str(dt):dict(metrics=aggregate([r['variants'][str(dt)]['metrics'] for r in records]),events=event_aggregate([r['variants'][str(dt)]['events'] for r in records]),
                         wall_ms=sum(r['variants'][str(dt)]['wall']['median_ms'] for r in records),device_ms=sum(r['variants'][str(dt)]['device']['median_ms'] for r in records)) for dt in configs}
    write(out/'complete.json',dict(finished=True,summary=summary,records=records,runtime=dict(repeats=a.repeats,shared=True,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),device=torch.cuda.get_device_name(),
        precision='FP32 IEEE',mask='Reference AP +/-10ms on common 1ms observation grid; events +/-5ms one-to-one.',timer='Prepared full CUDA Graph filter + quadratic/linear/bias + event reset, completion synchronized; external CPU packing/upload/download excluded.',manifest_sha256=sha(manifest))))
    print('GPU_COMPLETE',a.model,a.profile,summary,flush=True)
    publication=subprocess.run([sys.executable,str(work/'publish_evidence.py'),'--work',str(work),'--data',str(a.data.resolve())])
    if publication.returncode:write(out/'publication_error.json',dict(code=publication.returncode,experiment_complete=True,reason='Report postprocessing can be retried independently of the frozen experiment result.'))

if __name__=='__main__':main()
