"""Independent winner confirmation + transfer-inclusive wall time.

Tuning and confirming measurements are separated. No backend is selected using
NEURON fitting accuracy. The original R and F models are both timed again.
"""
import argparse,json,gc,time,sys
from pathlib import Path
import numpy as np,torch
from experiment import ROOT,HERE,case_data,measure_group,accuracy,stamp,resource_snapshot,CASES
from DBNN import DBNN_R,DBNN_F
from backends import Prepared,capture,subset_state

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch-only',action='store_true');args=ap.parse_args()
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    sweep=json.loads((HERE/'results_event.json').read_text())
    result={'started':stamp(),'resources_before':resource_snapshot(),'cases':[]}
    if args.batch_only:result=json.loads((HERE/'confirmation.json').read_text())
    for c in sweep['cases']:
        N,A,T,B=[c[k] for k in ['N','A','T','B']]
        if args.batch_only and B==1:continue
        sc,xc,ids,_=case_data(N,A,T,B);s={k:v.cuda() for k,v in sc.items()};x=xc.cuda()
        packed=xc[:,ids].contiguous().pin_memory();xs=packed.cuda();sp=subset_state(s,ids.cuda())
        backend=c['winner_exact'];tokens=backend.split('_');scan,quad=tokens[1],tokens[2]
        start=time.perf_counter();p=Prepared(sp,xs,scan,quad)
        fn=capture(p) if tokens[-1]=='graph' else p
        torch.cuda.synchronize();prepare_ms=(time.perf_counter()-start)*1000
        methods={'optimized':fn};models=[]
        for name,cls in [('pytorch_R',DBNN_R),('pytorch_F',DBNN_F)]:
            m=cls(N,T,torch.device('cuda'));m.load_state_dict(s);m.eval();models.append(m)
            methods[name]=lambda m=m:m(x)
        if B>1:
            for name,m in zip(['R','F'],models):
                methods[f'pytorch_{name}_trace_loop']=lambda m=m:torch.cat([m(x[b:b+1]) for b in range(B)])
        # Strong control: exact active-site packing with the unchanged PyTorch
        # equations, so the contribution of pruning and kernels is separable.
        for name,cls in [('pytorch_packed_R',DBNN_R),('pytorch_packed_F',DBNN_F)]:
            pm=cls(A,T,torch.device('cuda'));pm.load_state_dict(sp);pm.eval();models.append(pm)
            methods[name]=lambda pm=pm:pm(xs)
            if B>1:methods[name+'_trace_loop']=lambda pm=pm:torch.cat([pm(xs[b:b+1]) for b in range(B)])
        with torch.no_grad():
            ref=methods['pytorch_R']().clone()
            err=accuracy(fn().clone(),ref,p.vth)
            for f in methods.values():
                for _ in range(5):f()
            torch.cuda.synchronize()
            times=measure_group(methods,50)
            out_host=torch.empty((B,T),pin_memory=True)
            def e2e():
                xs.copy_(packed,non_blocking=True)
                output=fn()
                out_host.copy_(output,non_blocking=True)
                return out_host
            e2etimes=measure_group({'e2e':e2e},50)['e2e']
        names=[k for k in times if k.startswith('pytorch_') and 'packed' not in k];baseline=min(names,key=lambda k:times[k]['median_ms'])
        packed_best=min([k for k in times if k.startswith('pytorch_packed')],key=lambda k:times[k]['median_ms'])
        r={'index':c['index'],'N':N,'A':A,'T':T,'B':B,'backend':backend,
           'timings':times,'accuracy':err,'prepare_ms':prepare_ms,'best_original':baseline,
           'speedup':times[baseline]['median_ms']/times['optimized']['median_ms'],
           'packed_pytorch_best':packed_best,'kernel_speedup_over_packed':times[packed_best]['median_ms']/times['optimized']['median_ms'],
           'packed_raster_h2d_forward_d2h':e2etimes,
           'resources_after':resource_snapshot()}
        result['cases']=[old for old in result['cases'] if old['index']!=c['index']]+[r]
        result['cases'].sort(key=lambda c:c['index'])
        (HERE/'confirmation.json').write_text(json.dumps(result,indent=2))
        print('CONFIRM',c['index'],backend,round(times['optimized']['median_ms'],4),'ms',round(r['speedup'],2),'x',flush=True)
        del methods,models,p,fn,x,xs,s,sp,ref;gc.collect();torch.cuda.empty_cache()
    result['finished']=stamp();(HERE/'confirmation.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
