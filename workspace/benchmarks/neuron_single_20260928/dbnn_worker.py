"""Full latest DBNN pipeline, matched raster and complete timing boundaries."""
import argparse,gc,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
H=Path(__file__).resolve().parent;ROOT=H.parents[1];OLD=H.parent/'sparse_20260928'
sys.path[:0]=[str(OLD),str(ROOT/'source')]
from neuron_worker import inputs,stats,stamp
from DBNN import DBNN_R,DBNN_F
from backends import Prepared,capture,subset_state
from quad_optimized_backend import TiledPrepared
from tf32_pipeline import make_precision_prepared

def error(v,ref,threshold):
    v=v.detach().cpu();ref=ref.detach().cpu();d=v-ref
    return dict(rmse_mv=float(d.square().mean().sqrt()),max_abs_mv=float(d.abs().max()),reset_bin_difference=int(((v>=threshold)!=(ref>=threshold)).sum()),finite=bool(torch.isfinite(v).all()))

def measure(fns,n,gpu):
    vals={k:[] for k in fns};rng=random.Random(817)
    for fn in fns.values():
        for _ in range(3):fn()
    for _ in range(n):
        names=list(fns);rng.shuffle(names)
        for name in names:
            if gpu:torch.cuda.synchronize()
            t=time.perf_counter();fns[name]()
            if gpu:torch.cuda.synchronize()
            vals[name].append((time.perf_counter()-t)*1000)
    return {k:stats(v) for k,v in vals.items()}

@torch.no_grad()
def case(A,T,samples,rate,device,threads,repeats):
    ids,raw,_=inputs(A,T,samples,rate);B=len(samples);ii=torch.from_numpy(ids)
    packed=(torch.from_numpy(raw)*torch.where(ii<250,1.,-1.)[None,:,None]).contiguous()
    dense=torch.zeros(B,500,T);dense[:,ii]=packed
    state=torch.load(ROOT/'source/dbnn_checkpoint.pt',map_location='cpu',weights_only=False)['model_state']
    state={k:v.detach().clone() for k,v in state.items()}
    state['bilinear.weight'][0].fill_diagonal_(0)
    x=dense.to(device);s={k:v.to(device) for k,v in state.items()}
    xp=packed.to(device);sp=subset_state(s,ii.to(device));gpu=device=='cuda'
    models=[];fns={}
    for name,cls in [('original_R',DBNN_R),('original_F',DBNN_F)]:
        m=cls(500,T,torch.device(device));m.load_state_dict(s);m.eval();models.append(m)
        fns[name]=lambda m=m:m(x)
        if B>1:fns[name+'_trace_loop']=lambda m=m:torch.cat([m(x[b:b+1]) for b in range(B)])
    ref=fns['original_R']().clone();vth=float(s['v_th'])
    tune=[];prepare=time.perf_counter();pol=json.loads((OLD/'quad_backend_policy.json').read_text())
    if gpu:
        # New durations explicitly use/validate the latest fused kernel. No
        # unknown-shape cuBLAS fallback is labelled "all optimizations".
        spec=pol['shapes'][f'{A}:6001:{B}']['spec'] if f'{A}:6001:{B}' in pol['shapes'] else dict(backend='triton',config=3,triangle=A>32,precision='ieee')
        candidates={}
        for block in [64,128,256,512]:
            p=TiledPrepared(sp,xp,spec);p.block=block;fn=capture(p)
            er=error(fn().clone(),ref,vth)
            assert er['finite'] and er['max_abs_mv']<1e-3 and er['reset_bin_difference']==0,er
            candidates[str(block)]=fn
        screen=measure(candidates,10,True)
        block=min(screen,key=lambda k:screen[k]['median_ms']);fn=candidates[block];p=fn.prepared
        tune=[dict(scan_block=int(k),**v) for k,v in screen.items()]
        fns['optimized_FP32']=fn
        # Graph must read changed input; restore it before measurements.
        xp.zero_();zero=fn().clone();torch.cuda.synchronize()
        assert float((zero-p.bias).abs().max())<1e-4
        xp.copy_(packed);torch.cuda.synchronize()
        policy=json.loads((OLD/'tf32_backend_policy.json').read_text())
        if f'{A}:{T}:{B}' in policy['profiles']['strict']['shapes']:
            pp=make_precision_prepared(sp,xp,policy,'strict');fp=capture(pp)
            er=error(fp().clone(),ref,vth)
            if er['finite'] and er['max_abs_mv']<1e-3 and er['reset_bin_difference']==0:fns['precision_strict']=fp
        implementation=dict(active_set=A,scan='Triton prefix scan',scan_block=int(block),drive_spec=spec,reset='GPU event search',schedule='cached coefficients/workspaces + CUDA Graph',all_five_layers=True)
    else:
        p=Prepared(sp,xp,'cpu','cublas');fns['optimized_CPU']=p
        implementation=dict(active_set=A,scan='native recurrence across site/batch',drive='CPU BLAS + reduction',reset='native CPU recurrence',threads=threads,all_five_layers=False)
    if gpu:torch.cuda.synchronize()
    prepare_ms=(time.perf_counter()-prepare)*1000
    errs={k:error(f().clone(),ref,vth) for k,f in fns.items()}
    out=fns['optimized_FP32' if gpu else 'optimized_CPU']().clone().cpu().numpy()
    core=measure(fns,repeats,gpu)
    main=fns['optimized_FP32' if gpu else 'optimized_CPU']
    # Full signed dense CPU raster is the boundary. Known active IDs are reused.
    packed_buffer=torch.empty(packed.shape,pin_memory=gpu)
    host_out=torch.empty((B,T),pin_memory=gpu)
    def packed_transfer():
        xp.copy_(packed_buffer,non_blocking=gpu)
        host_out.copy_(main(),non_blocking=gpu)
        return host_out
    def e2e():
        torch.index_select(dense,1,ii,out=packed_buffer)
        return packed_transfer()
    packed_buffer.copy_(packed)
    transfer=measure({'packed_h2d_forward_d2h':packed_transfer,'dense_gather_h2d_forward_d2h':e2e},repeats,gpu)
    events=[]
    if gpu:
        e1=torch.cuda.Event(enable_timing=True);e2=torch.cuda.Event(enable_timing=True)
        for _ in range(repeats):
            e1.record();main();e2.record();e2.synchronize();events.append(float(e1.elapsed_time(e2)))
    result=dict(A=A,T=T,B=B,samples=samples,rate=rate,device=device,threads=threads,input_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                implementation=implementation,shape_screen=tune,prepare_and_screen_ms=prepare_ms,accuracy_vs_original=errs,timing=core,transfer=transfer,
                cuda_event=stats(events) if events else None,loadavg=os.getloadavg(),affinity=sorted(os.sched_getaffinity(0)))
    if gpu:result['gpu_state']=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader'],text=True)
    del fns,models,ref,x,xp,s,sp,p;gc.collect()
    if gpu:torch.cuda.empty_cache()
    return result,out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda');ap.add_argument('--threads',type=int,default=1);ap.add_argument('--repeats',type=int,default=40)
    ap.add_argument('--output',default='dbnn_gpu.json');a=ap.parse_args()
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    cs=[(A,T,[0],'sampled') for T in [1001,6001] for A in [32,128,500]]
    if a.device=='cuda':cs += [(500,1001,[0],r) for r in ['quiet','low','high']]+[(500,6001,[0,1,2],'sampled'),(500,1001,list(range(8)),'sampled')]
    else:cs=[(A,6001,[0],'sampled') for A in ([32,128,500] if a.threads==1 else [500])]
    out=H/a.output;out.parent.mkdir(exist_ok=True)
    report=json.loads(out.read_text()) if out.exists() else dict(started=stamp(),torch=torch.__version__,precision='FP32; explicit strict profile listed separately',cases=[])
    for A,T,samples,rate in cs:
        if any((r['A'],r['T'],r['samples'],r['rate'])==(A,T,samples,rate) for r in report['cases']):continue
        r,v=case(A,T,samples,rate,a.device,a.threads,a.repeats)
        stem=f'dbnn_{a.device}_n{a.threads}_A{A}_T{T}_B{len(samples)}_{rate}'
        np.savez_compressed(H/'results'/f'{stem}.npz',voltage=v)
        r['voltage_file']=stem+'.npz';report['cases'].append(r)
        out.write_text(json.dumps(report,indent=2));print('DBNN',stem,{k:round(v['median_ms'],4) for k,v in r['timing'].items()},flush=True)
    report['finished']=stamp();out.write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
