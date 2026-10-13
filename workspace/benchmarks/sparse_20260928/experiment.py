"""Matched frozen-checkpoint and synthetic sparse-site scalability experiments."""
import argparse,hashlib,json,os,random,sys,time,traceback,gc,subprocess
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'source'))
sys.path.insert(0,str(ROOT/'scripts'))
from DBNN import DBNN_R,DBNN_F
from backends import Prepared,capture,subset_state

CASES=[
    (500,500,6001,1),(500,32,6001,1),(500,128,6001,1),(500,8,6001,1),
    (500,500,6001,8),(500,32,6001,8),
    (500,32,1001,1),(2000,32,1001,1),(8000,32,1001,1),
    (500,128,1001,1),(2000,128,1001,1),(8000,128,1001,1),
    (2000,128,6001,1),(500,32,301,1),
]

def stamp():return time.strftime('%Y-%m-%dT%H:%M:%S%z')
def resource_snapshot():
    r={'time':stamp(),'cpu_affinity':sorted(os.sched_getaffinity(0)), 'loadavg':os.getloadavg()}
    for key,args in [('gpus',['--query-gpu=index,uuid,utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader']),
                     ('processes',['--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader'])]:
        r[key]=subprocess.check_output(['nvidia-smi']+args,text=True).strip()
    return r

def case_data(N,A,T,B):
    ck=torch.load(ROOT/'source/dbnn_checkpoint.pt',map_location='cpu',weights_only=False)
    raw=ck['model_state']
    if N==500:mp=torch.arange(N)
    else:mp=torch.arange(N)%250+(torch.arange(N)>=N//2).long()*250
    state={k:v.detach().clone() for k,v in raw.items()}
    for k in ('tau_rise','tau_decay','omega'):state[k]=state[k][mp]
    state['bilinear.weight']=state['bilinear.weight'][:,mp][:,:,mp].contiguous()
    state['bilinear.weight'][0].fill_diagonal_(0)
    rng=np.random.default_rng(918+A+N)
    ids=torch.as_tensor(np.sort(rng.choice(N,A,replace=False)),dtype=torch.long)
    d=np.load(ROOT/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True)
    orig=torch.from_numpy(d['test_ip'][:B,:, :T].astype(np.float32))
    signs=torch.where(mp<250,1.,-1.)
    x=torch.zeros((B,N,T),dtype=torch.float32)
    x[:,ids]=orig[:,mp[ids]]*signs[ids,None]
    return state,x,ids,mp

def stats(ts):
    return {'median_ms':float(np.median(ts)),'mean_ms':float(np.mean(ts)),
            'std_ms':float(np.std(ts,ddof=1)) if len(ts)>1 else None,
            'p95_ms':float(np.percentile(ts,95)),'n':len(ts),'runs_ms':ts}

def accuracy(out,ref,vth):
    out=out.detach().cpu();ref=ref.detach().cpu();e=out-ref
    return {'rmse_mv':float(e.square().mean().sqrt()),'max_abs_mv':float(e.abs().max()),
       'spike_mismatch_bins':int(((out>=vth)!=(ref>=vth)).sum()),'finite':bool(torch.isfinite(out).all())}

def measure_group(fns,reps,gpu=True):
    values={name:[] for name in fns}
    rng=random.Random(917)
    for i in range(reps):
        order=list(fns);rng.shuffle(order)
        for name in order:
            if gpu:torch.cuda.synchronize()
            start=time.perf_counter();out=fns[name]()
            if gpu:torch.cuda.synchronize()
            values[name].append((time.perf_counter()-start)*1000)
    return {name:stats(ts) for name,ts in values.items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=HERE/'results.json')
    ap.add_argument('--case-start',type=int,default=0);ap.add_argument('--case-stop',type=int,default=len(CASES))
    ap.add_argument('--repeats',type=int,default=15);ap.add_argument('--cpu-repeats',type=int,default=7)
    ap.add_argument('--skip-cpu',action='store_true');ap.add_argument('--resume',action='store_true')
    args=ap.parse_args();torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    report={'created':stamp(),'scope':'core_forward; packed IDs/W are preparation, reported separately',
      'torch':torch.__version__,'cuda':torch.version.cuda,'fp32_matmul_tf32':False,
      'baseline_sha256':hashlib.sha256((ROOT/'source/DBNN.py').read_bytes()).hexdigest(),
      'checkpoint_sha256':hashlib.sha256((ROOT/'source/dbnn_checkpoint.pt').read_bytes()).hexdigest(),
      'resources_before':resource_snapshot(),'cases':[],'attempt_errors':[]}
    if args.resume and args.output.exists():report=json.loads(args.output.read_text())
    for ci in range(args.case_start,args.case_stop):
        if any(c['index']==ci for c in report['cases']):continue
        N,A,T,B=CASES[ci];print(f'CASE {ci}: N={N} A={A} T={T} B={B}',flush=True)
        sc,xc,ids,mp=case_data(N,A,T,B)
        s={k:v.cuda() for k,v in sc.items()};x=xc.cuda();idg=ids.cuda()
        torch.cuda.synchronize();tp=time.perf_counter()
        xs=x.index_select(1,idg).contiguous();sp=subset_state(s,idg)
        torch.cuda.synchronize();packing_ms=(time.perf_counter()-tp)*1000
        rec={'index':ci,'N':N,'A':A,'T':T,'B':B,'duration_ms':T-1,
             'kind':'trained_500_site_checkpoint' if N==500 else 'synthetic_parameter_expansion_not_retrained',
             'event_count':int((xc!=0).sum()),'packing_once_ms':packing_ms,
             'input_sha256':hashlib.sha256(xc.numpy().tobytes()).hexdigest(),
             'methods':{},'resources':resource_snapshot()}
        refs={};fns={};owners=[]
        with torch.no_grad():
            for name,cls in [('pytorch_R_full',DBNN_R),('pytorch_F_full',DBNN_F)]:
                m=cls(N,T,torch.device('cuda'));m.load_state_dict(s);m.eval()
                fn=lambda m=m: m(x)
                out=fn().clone();torch.cuda.synchronize()
                owners.append(m);fns[name]=fn
                if name=='pytorch_R_full':ref=out.clone()
                refs[name]=accuracy(out,ref,float(s['v_th'].item()))
            # Best public original implementation may process each sample via its B=1 fast path.
            if B>1:
                for name,m in zip(['R','F'],owners):
                    fn=lambda m=m:torch.cat([m(x[b:b+1]) for b in range(B)])
                    fns[f'pytorch_{name}_single_trace_loop']=fn
                    refs[f'pytorch_{name}_single_trace_loop']=accuracy(fn(),ref,float(s['v_th']))
            variants=[('cuda','cublas'),('cuda','direct'),('cuda','cutlass'),('triton','cublas'),
                      ('fft','cublas'),('fft','cutlass'),('cute','cublas')]
            for scan,quad in variants:
                name=f'packed_{scan}_{quad}'
                try:
                    p=Prepared(sp,xs,scan,quad);out=p().clone();torch.cuda.synchronize()
                    er=accuracy(out,ref,p.vth)
                    fns[name+'_eager']=p;refs[name+'_eager']=er
                    graph=capture(p)
                    # Critical validation: captured graph must read the new input, not stale output.
                    saved=xs.clone();xs.zero_();z=graph().clone();torch.cuda.synchronize()
                    if float((z-p.bias).abs().max())>1e-5:raise AssertionError('Graph failed zero-input check')
                    xs.copy_(saved);out=graph().clone();torch.cuda.synchronize()
                    er_graph=accuracy(out,ref,p.vth)
                    fns[name+'_graph']=graph;refs[name+'_graph']=er_graph
                except Exception as e:
                    report['attempt_errors'].append({'case':ci,'method':name,'error':str(e)})
                    print(name,'ERROR',str(e)[:600],flush=True)
            # Tensor-core approximation is assessed separately, against full FP32 voltage/spikes.
            for quad in ['tf32','bf16','fp16']:
                try:
                    p=Prepared(sp,xs,'cuda',quad);fn=capture(p)
                    name=f'packed_cuda_{quad}_graph';fns[name]=fn;refs[name]=accuracy(fn().clone(),ref,p.vth)
                except Exception as e:report['attempt_errors'].append({'case':ci,'method':quad,'error':str(e)})
            # Warm all paths before randomized interleaved timing.
            for fn in fns.values():
                for _ in range(3):fn()
            torch.cuda.synchronize()
            timing=measure_group(fns,args.repeats)
            for name,value in timing.items():
                rec['methods'][name]={**value,**refs[name],'device':'cuda:0',
                     'throughput_traces_s':1000*B/value['median_ms']}
            eligible=[name for name in fns if name.startswith('packed') and not any(q in name for q in ('bf16','fp16','tf32')) and refs[name]['finite'] and refs[name]['max_abs_mv']<1e-3 and refs[name]['spike_mismatch_bins']==0]
            best=min(eligible,key=lambda name:timing[name]['median_ms'])
            rec['winner_exact']=best
            baseline=[k for k in timing if k.startswith('pytorch_')]
            rec['baseline_best']=min(baseline,key=lambda name:timing[name]['median_ms'])
            rec['speedup_vs_best_original']=timing[rec['baseline_best']]['median_ms']/timing[best]['median_ms']
            # Input changes/packing are measured explicitly; no graph rebuilding in steady state.
            chosen=fns[best]
            def with_gather():
                torch.index_select(x,1,idg,out=xs)
                return chosen()
            rec['winner_with_input_gather']=measure_group({'gather':with_gather},args.repeats)['gather']
            # Wall-clock is primary; events add stream elapsed time under current contention.
            ev1=torch.cuda.Event(enable_timing=True);ev2=torch.cuda.Event(enable_timing=True)
            et=[]
            for _ in range(args.repeats):
                ev1.record();chosen();ev2.record();ev2.synchronize();et.append(float(ev1.elapsed_time(ev2)))
            rec['winner_cuda_events']=stats(et)
            print('WINNER',best,round(timing[best]['median_ms'],4),'ms',round(rec['speedup_vs_best_original'],2),'x',flush=True)
            if not args.skip_cpu:
                cpu_fns={};cpu_owners=[]
                for name,cls in [('pytorch_R_cpu',DBNN_R),('pytorch_F_cpu',DBNN_F)]:
                    cm=cls(N,T,torch.device('cpu'));cm.load_state_dict(sc);cm.eval()
                    cpu_fns[name]=lambda cm=cm:cm(xc);cpu_owners.append(cm)
                cp=Prepared(subset_state(sc,ids),xc[:,ids].contiguous(),'cpu','cublas')
                cpu_fns['packed_native_cpu']=cp
                cpu_err={name:accuracy(fn(),ref,float(sc['v_th'])) for name,fn in cpu_fns.items()}
                for fn in cpu_fns.values():fn()
                times=measure_group(cpu_fns,args.cpu_repeats,gpu=False)
                for name,value in times.items():rec['methods'][name]={**value,**cpu_err[name],'device':'cpu','threads':8}
            rec['resources_after']=resource_snapshot()
            report['cases'].append(rec);args.output.write_text(json.dumps(report,indent=2))
        # Release graph pools and per-case storage before the next large dense baseline.
        del fns,owners,refs,s,x,xs,sp,ref;gc.collect();torch.cuda.empty_cache()
    report['finished']=stamp();report['resources_after']=resource_snapshot()
    args.output.write_text(json.dumps(report,indent=2));print('COMPLETE',args.output,flush=True)

if __name__=='__main__':main()
