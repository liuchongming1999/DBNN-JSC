"""Full prepared inference versus the previous GPU implementation, with separate I/O.

The main latency is one invocation, with device synchronization, not amortized
over a replay batch. Inputs, coefficients and workspaces are ready before it.
"""
import argparse,gc,hashlib,os,random,subprocess,sys,time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

def stat(x):
    return dict(median_ms=float(np.median(x)),q1_ms=float(np.quantile(x,.25)),q3_ms=float(np.quantile(x,.75)),runs_ms=x)

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--repeats',type=int,default=30)
    a=ap.parse_args();root=a.root.resolve();work=a.work.resolve();dtwork=root/'benchmarks/dt_resolution_20260929'
    sys.path[:0]=[str(root/'benchmarks/sparse_20260928'),str(dtwork)]
    from common import read,write,sha
    from backends import subset_state
    from backend import Model,Graph,numerical
    from benchmark_dt import variants,checks
    os.sched_setaffinity(0,{72});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    policy=read(work/'policy.json');assert policy['finished'] and len(policy['records'])==24
    # The control's expanded FP32 raster is prepared before compute timing.
    # Deployment must accept the compact signed raster without precomputing it.
    for q in policy['records']:
        valid=[n for n in q['timing'] if not n.startswith('baseline') and q['validation'][n]['passed']]
        q['selected']=min(valid,key=lambda n:q['timing'][n]['device_ms'])
    policy['rule']='Fastest valid compact-input implementation on independent synthetic 25Hz data; same A and nearest log input length. Baseline measured as control, excluded from deployment selection because it uses a prepared expanded raster.'
    write(work/'deployment_policy.json',policy)
    specs=variants(root,dtwork);states={n:torch.load(p,map_location='cuda',weights_only=False)['model_state'] for n,(_,p) in specs.items()}
    source=read(dtwork/'test/manifest.json')['cases'];out=work/'gpu';out.mkdir(exist_ok=True)
    rng=random.Random(382947);records=[]
    runtime=dict(started=datetime.now().isoformat(),device=torch.cuda.get_device_name(),gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),shared=True,
        resource_start=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'],capture_output=True,text=True).stdout,
        repeats=a.repeats,cpu_affinity=[72],precision='FP32 IEEE; lossless int8 event input; FP32 accumulation',
        wall_scope='One complete CUDA Graph replay followed by torch.cuda.synchronize; input already device-resident. Preceding synchronization is outside timer.',
        device_scope='External timing CUDA events captured inside the graph around the entire filter+drive+reset forward.',
        e2e_scope='CPU signed compact dense raster -> cached pinned active-site gather -> H2D -> complete compute -> D2H -> synchronize. All setup and allocations excluded.',
        input_scope='Integer-ms event amplitudes stored losslessly as signed int8. Event times remain identical at both output intervals. Half-ms zero input is implicit.',
        variants={n:dict(dt_ms=dt,path=str(p),checkpoint_sha256=sha(p)) for n,(dt,p) in specs.items()})
    write(out/'runtime.json',runtime)
    for row in source:
        cid=row['case_id'];rawfile=np.load(row['input_file']);ids=rawfile['ids'].astype(np.int64);raw=rawfile['x']
        assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256']
        assert np.equal(raw,np.round(raw)).all() and raw.min()>=0 and raw.max()<=127
        signed=raw.astype(np.int8)*(1-2*(ids[:,None]>=250)).astype(np.int8)
        ii=torch.from_numpy(ids);packed=torch.from_numpy(signed)[None].contiguous();Tin=packed.shape[-1]
        x=packed.cuda();dense=torch.zeros((1,500,Tin),dtype=torch.int8,pin_memory=True);dense[:,ii]=packed
        identity=bool(np.array_equal(ids,np.arange(500)))
        gather=dense if identity else torch.empty_like(packed,device='cpu',pin_memory=True)
        holders={};predictions={};drives={};validation={};selected={};nearby={}
        for name,(dt,p) in specs.items():
            near=min([q for q in policy['records'] if q['A']==len(ids) and q['dt_ms']==dt],key=lambda q:abs(np.log(q['input_steps']/Tin)))
            mode,block=near['selected'].rsplit('_',1);selected[name]=near['selected'];nearby[name]=near['input_steps']
            state=subset_state(states[name],ii.cuda());oracle_np=np.load(dtwork/'gpu_optimized'/name/(cid+'.npz'))['voltage']
            oracle=torch.from_numpy(oracle_np).cuda()[None]
            for tag,mm,bb in [('baseline','baseline',512),('optimized',mode,int(block))]:
                key=(name,tag);model=Model(state,x,dt,mm,bb);v=model();torch.cuda.synchronize()
                val=numerical(v,oracle,model.th)
                if not val['passed']:raise RuntimeError((cid,key,'vs independently validated old result',val))
                validation[key]=val;predictions[key]=v[0].cpu().numpy().copy();drives[key]=model.drive[0].cpu().numpy().copy()
                holders[key]=dict(model=model,plain=Graph(model),events=Graph(model,events=True))
                if tag=='optimized':holders[key]['host']=torch.empty_like(model.out,device='cpu',pin_memory=True)
        walls={k:[] for k in holders};devices={k:[] for k in holders};e2e={n:[] for n in specs}
        def pack():
            if not identity:torch.index_select(dense,1,ii,out=gather)
        def end_to_end(h):
            pack();x.copy_(gather,non_blocking=True);h['host'].copy_(h['plain'](),non_blocking=True);torch.cuda.synchronize()
        for h in holders.values():
            for _ in range(3):h['plain']()
        torch.cuda.synchronize()
        for _ in range(a.repeats):
            keys=list(holders);rng.shuffle(keys)
            for key in keys:
                h=holders[key];torch.cuda.synchronize();tick=time.perf_counter();h['plain']();torch.cuda.synchronize()
                walls[key].append((time.perf_counter()-tick)*1000)
                h['events']();torch.cuda.synchronize();devices[key].append(h['events'].elapsed_ms())
            names=list(specs);rng.shuffle(names)
            for name in names:
                tick=time.perf_counter();end_to_end(holders[(name,'optimized')]);e2e[name].append((time.perf_counter()-tick)*1000)
        rec=dict(case_id=cid,A=len(ids),input_steps=Tin,duration_s=row['duration_ms']/1000,input_sha256=row['input_sha256'],variants={})
        for name,(dt,p) in specs.items():
            key=(name,'optimized');h=holders[key];model=h['model']
            hostcheck=checks(h['host'][0].numpy(),predictions[key],model.th)
            if not hostcheck['passed']:raise RuntimeError(('timed_host_output',cid,name,hostcheck))
            component={}
            for comp,fn in [('filter',model.filter),('drive',model.integrate),('reset',model.reset)]:
                gr=Graph(fn,events=True);times=[]
                for _ in range(10):gr();torch.cuda.synchronize();times.append(gr.elapsed_ms())
                component[comp]=stat(times);del gr
            io={t:[] for t in ['pack','h2d_wall','d2h_wall']}
            for _ in range(10):
                tick=time.perf_counter();pack();io['pack'].append((time.perf_counter()-tick)*1000)
                tick=time.perf_counter();x.copy_(gather,non_blocking=True);torch.cuda.synchronize();io['h2d_wall'].append((time.perf_counter()-tick)*1000)
                tick=time.perf_counter();h['host'].copy_(model.out,non_blocking=True);torch.cuda.synchronize();io['d2h_wall'].append((time.perf_counter()-tick)*1000)
            ev=np.flatnonzero(predictions[key]>=model.th)*dt
            (out/name).mkdir(exist_ok=True);np.savez_compressed(out/name/(cid+'.npz'),voltage=predictions[key],drive=drives[key],reset_events_ms=ev)
            rec['variants'][name]=dict(dt_ms=dt,checkpoint_sha256=sha(p),backend=selected[name],policy_input_steps=nearby[name],
                optimized=dict(wall=stat(walls[key]),device=stat(devices[key]),validation=validation[key]),
                baseline=dict(wall=stat(walls[(name,'baseline')]),device=stat(devices[(name,'baseline')]),validation=validation[(name,'baseline')]),
                e2e=stat(e2e[name]),io={k:stat(v) for k,v in io.items()},components_device=component,
                timed_path_validation=hostcheck,input_bytes=int(x.numel()),output_bytes=int(model.out.numel()*4),reset_events_ms=ev.tolist())
        records.append(rec);write(out/(cid+'.json'),rec);write(out/'status.json',dict(completed=len(records),total=len(source),latest=cid))
        print('BENCH',cid,{n:dict(core=round(rec['variants'][n]['optimized']['wall']['median_ms'],4),device=round(rec['variants'][n]['optimized']['device']['median_ms'],4),e2e=round(np.median(e2e[n]),4),backend=selected[n]) for n in specs},flush=True)
        del holders,h,model,x,dense,gather,oracle,packed,state;gc.collect();torch.cuda.empty_cache()
    runtime['resource_end']=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'],capture_output=True,text=True).stdout
    write(out/'complete.json',dict(finished=True,records=records,runtime=runtime));print('COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
