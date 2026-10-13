"""Prepared B=1 inference timing, locked weights, numerical oracle per input."""
import argparse,gc,json,os,sys,time,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch


def read(p):return json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def stat(a):return dict(median_ms=float(np.median(a)),q1_ms=float(np.quantile(a,.25)),q3_ms=float(np.quantile(a,.75)),runs_ms=a)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(v,oracle,ev,threshold):
    v=np.asarray(v);ev2=np.flatnonzero(v>=threshold).astype(float)
    r=dict(max_abs_mv=float(np.max(np.abs(v-oracle))),rmse_mv=float(np.sqrt(np.mean((v.astype(float)-oracle)**2))),events_equal=bool(np.array_equal(ev2,ev)),finite=bool(np.isfinite(v).all()))
    r['passed']=r['finite'] and r['max_abs_mv']<.001 and r['events_equal'];return r


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True)
    p.add_argument('--method',choices=['gpu','cpu1','cpu8','tcn','elm','lstm'],required=True);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();root=a.root.resolve();panel=a.work.resolve();base=panel/'main379';method=a.method;out=base/method;out.mkdir(exist_ok=True)
    rows=read(base/'manifest.json')['cases'];assert len(rows)==379
    sys.path[:0]=[str(root/'benchmarks/compute_fair_20260929'),str(root/'benchmarks/sparse_20260928'),str(root/'benchmarks/fair_window_20261005/timing_w500'),
                  str(root/'benchmarks/cpu_20260929'),str(root/'benchmarks/baselines_20260930'),str(root/'benchmarks/baselines_20260930/deps')]
    nt=8 if method=='cpu8' else 1;cpus=list(range(80,80+nt)) if method.startswith('cpu') else [{'gpu':72,'tcn':73,'elm':74,'lstm':75}[method]]
    os.sched_setaffinity(0,cpus);torch.set_num_threads(nt);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from gpu_backend import ResolutionModel,Graph
    from backends import subset_state
    from cpu_backend import CPUModel
    from benchmark_cpu import methods
    if method in ['gpu','cpu1','cpu8']:
        cp=Path(read(panel/'confirmatory/manifest.json')['checkpoint']);state=torch.load(cp,map_location='cuda' if method=='gpu' else 'cpu',weights_only=False)['model_state']
        assert torch.diagonal(state['bilinear.weight'][0]).abs().max()>0
        policy=read(root/f'benchmarks/fair_window_20261005/timing_w500/{"gpu_policy.json" if method=="gpu" else method+"/policy.json"}')['records']
    else:
        src=root/'benchmarks/baselines_20260930';lock=read(src/'aligned'/f'{method}_1ms/complete.json');cfg=lock['config'];cp=src/'aligned'/f'{method}_1ms/best.pt'
        assert sha(cp)==lock['checkpoint_sha256'];cal=lock['validation']['calibration']
        from train_default import model_for,predict_case
        model=model_for(src,cfg).cuda().eval();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'])
    runtime=dict(method=method,checkpoint=str(cp),checkpoint_sha256=sha(cp),repeats=a.repeats,cpus=cpus,shared_server=True,
                 scope='One prepared complete forward to voltage; learned baselines also output sigmoid spike scores. Includes synchronization. External raster packing, upload/download, model preparation, capture and scoring excluded.',
                 precision='FP32 IEEE (GPU); FP32/FMA and MKL (CPU)',gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 device=torch.cuda.get_device_name() if not method.startswith('cpu') else None,
                 resources_start=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'],capture_output=True,text=True).stdout,
                 policy='Reuse the preceding synthetic shape policy; numerical fallback only, never accuracy-based checkpoint selection',script_sha256=sha(__file__))
    write(out/'runtime.json',runtime);records=[];graph_cache={}
    for row in rows:
        cid=row['case_id'];cached=out/f'{cid}.json'
        if cached.exists():
            rec=read(cached);assert rec['checkpoint_sha256']==sha(cp) and rec['input_sha256']==row['input_sha256'];records.append(rec);continue
        d=np.load(row['input_file']);raw=d['x'];ids=d['ids'];assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256']
        validation=None;dev=[];times=[];graph=None
        if method=='gpu':
            old=np.load(panel/'predictions'/f'{cid}.npz');x=torch.from_numpy(raw.astype(np.int8)*np.where(ids[:,None]<250,1,-1).astype(np.int8))[None].contiguous().cuda()
            st=subset_state(state,torch.as_tensor(ids.astype(np.int64),device='cuda'));near=min([c for c in policy if c['dt_ms']==1. and c['A']==len(ids)],key=lambda c:abs(np.log(c['input_steps']/x.shape[-1])))
            candidates=sorted(near['timing'],key=lambda k:near['timing'][k]['median_ms'])
            for tag in candidates:
                mode,block=tag.rsplit('_',1);m=ResolutionModel(st,x,1.,mode,int(block));v=m();torch.cuda.synchronize()
                validation=check(v[0].cpu().numpy(),old['voltage'],old['predicted_events_ms'],m.th)
                if validation['passed']:break
                del m,v
            else:raise RuntimeError((cid,method,validation))
            graph=Graph(m);device_graph=Graph(m,events=True)
            for _ in range(a.repeats):
                torch.cuda.synchronize();start=time.perf_counter();graph();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
                device_graph();torch.cuda.synchronize();dev.append(device_graph.elapsed_ms())
            validation=check(m.out[0].cpu().numpy(),old['voltage'],old['predicted_events_ms'],m.th);assert validation['passed']
            del graph,device_graph,m,v,x,st;graph=None
        elif method.startswith('cpu'):
            old=np.load(panel/'predictions'/f'{cid}.npz');dense=np.zeros((500,row['duration_ms']+1),np.float32);dense[ids]=raw*np.where(ids[:,None]<250,1,-1)
            m=CPUModel(state,ids.astype(np.int64),dense.shape[1]);near=min([c for c in policy if c['dt_ms']==1. and c['A']==len(ids)],key=lambda c:abs(np.log(c['T']/dense.shape[1])))
            configs=methods();candidates=sorted(near['timing'],key=lambda k:near['timing'][k]['wall']['median_ms'])
            for tag in candidates:
                validation=check(m.run(dense,nt,**configs[tag]),old['voltage'],old['predicted_events_ms'],m.threshold)
                if validation['passed']:break
            else:raise RuntimeError((cid,method,validation))
            for _ in range(3):m.run(dense,nt,**configs[tag])
            for _ in range(a.repeats):start=time.perf_counter();m.run(dense,nt,**configs[tag]);times.append((time.perf_counter()-start)*1000)
            assert check(m.output,old['voltage'],old['predicted_events_ms'],m.threshold)['passed'];m.close()
        else:
            dense=np.zeros((row['duration_ms']+1,500),np.float32);dense[:,ids]=raw.T
            if method!='tcn':dense[:,250:]*=-1
            cache=graph_cache.get(len(dense))
            inp=torch.from_numpy(dense).cuda() if cache is None else cache['input']
            if cache is not None:inp.copy_(torch.from_numpy(dense))
            def forward(inp=inp):
                y=predict_case(model,inp,cfg,'cuda');v=y[:,1]*(10 if method!='tcn' else 1)-67.7
                return v*cal['scale']+cal['bias_mv'],torch.sigmoid(y[:,0])
            if cache is None:eager=forward();torch.cuda.synchronize()
            else:
                graph=cache['graph'];captured=cache['captured'];graph.replay();torch.cuda.synchronize();eager=captured
            key={'tcn':'tcn_author_default','elm':'elm_author_default','lstm':'lstm_common150'}[method]
            old=np.load(panel/'methods'/key/'predictions'/f'{cid}.npz')
            gap=max(float(np.max(np.abs(eager[0].cpu().numpy()-old['voltage']))),float(np.max(np.abs(eager[1].cpu().numpy()-old['probability']))));assert gap<1e-4
            failure=None
            if cache is None:
                for _ in range(3):eager=forward()
                torch.cuda.synchronize()
                try:
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):captured=forward()
                    graph.replay();torch.cuda.synchronize()
                    gap2=max(float(torch.max(torch.abs(x-y))) for x,y in zip(captured,eager));assert gap2<1e-4
                    graph_cache[len(dense)]={'input':inp,'graph':graph,'captured':captured}
                except Exception as e:graph=None;failure=repr(e)
            tag='CUDA Graph' if graph is not None else 'eager'
            for _ in range(a.repeats):
                torch.cuda.synchronize();start=time.perf_counter()
                if graph is not None:graph.replay()
                else:eager=forward()
                torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            validation=dict(passed=True,voltage_and_probability_max_gap=gap,capture_failure=failure)
            del graph,eager,inp;graph=None
        rec=dict(case_id=cid,duration_s=row['duration_s'],A=row['A'],input_sha256=row['input_sha256'],checkpoint_sha256=sha(cp),
                 wall=stat(times),device=stat(dev) if dev else None,backend=tag,validation=validation)
        write(cached,rec);records.append(rec);write(out/'status.json',dict(completed=len(records),total=len(rows)))
        if len(records)%20==0:print(method,len(records),len(rows),flush=True)
        gc.collect()
        if not method.startswith('cpu'):torch.cuda.empty_cache()
    write(out/'complete.json',dict(finished=True,method=method,records=records,wall_ms=sum(r['wall']['median_ms'] for r in records),runtime=runtime))
    print('COMPLETE',method,sum(r['wall']['median_ms'] for r in records),flush=True)


if __name__=='__main__':main()
