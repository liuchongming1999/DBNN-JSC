import argparse,gc,os,random,sys,time
from pathlib import Path
import numpy as np
import torch

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--work',type=Path,required=True);ap.add_argument('--pilot',action='store_true')
    a=ap.parse_args();root=a.root.resolve();work=a.work.resolve();work.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(root/'benchmarks/sparse_20260928'),str(root/'benchmarks/dt_resolution_20260929')]
    from backends import subset_state
    from backend import Model,Graph,options,numerical
    from common import write,read,sha
    os.sched_setaffinity(0,{72});torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    p=root/'benchmarks/channel_best_pipeline_20260929/final_main/best_checkpoint.pt';state=torch.load(p,map_location='cuda',weights_only=False)['model_state']
    shapes=[(64,1001,.5),(500,16001,.5),(500,16001,1.)] if a.pilot else [(K,T,dt) for K in [64,128,256,500] for T in [1001,4001,16001] for dt in [1.,.5]]
    records=[];order=random.Random(82933)
    for K,Tin,dt in shapes:
        gen=torch.Generator(device='cuda').manual_seed(99227+K+Tin)
        ids=torch.randperm(500,generator=gen,device='cuda')[:K].sort()[0];s=subset_state(state,ids)
        x=(torch.rand((1,K,Tin),generator=gen,device='cuda')<.025).to(torch.int8)*(1-2*(ids>=250).to(torch.int8))[None,:,None]
        base=Model(s,x,dt,'baseline',512)
        expected=base().clone();ey=base.y.clone();eu=base.drive.clone();torch.cuda.synchronize()
        models={};graphs={};validation={};samples={};host={}
        for mode,block in options(dt):
            name=f'{mode}_{block}';m=Model(s,x,dt,mode,block)
            pred=m();torch.cuda.synchronize();v=numerical(pred,expected,m.th)
            v['filter_max_abs']=float((m.y-ey).abs().max());v['drive_max_abs_mv']=float((m.drive-eu).abs().max())
            validation[name]=v
            if not v['passed']:print('INVALID',K,Tin,dt,name,v,flush=True);continue
            models[name]=m;graphs[name]=Graph(m,events=True);samples[name]=[];host[name]=[]
        for _ in range(15 if not a.pilot else 7):
            names=list(graphs);order.shuffle(names)
            for name in names:
                torch.cuda.synchronize();start=time.perf_counter();graphs[name]();torch.cuda.synchronize()
                host[name].append((time.perf_counter()-start)*1000);samples[name].append(graphs[name].elapsed_ms())
        selected=min(samples,key=lambda k:np.median(samples[k]))
        rec=dict(A=K,input_steps=Tin,output_steps=base.T,dt_ms=dt,selected=selected,validation=validation,
            timing={n:dict(device_ms=float(np.median(samples[n])),wall_ms=float(np.median(host[n])),device_runs=samples[n],wall_runs=host[n]) for n in samples})
        records.append(rec);write(work/('pilot.json' if a.pilot else 'policy.json'),dict(finished=False,records=records))
        print('TUNED',K,Tin,dt,selected,{n:round(float(np.median(v)),5) for n,v in samples.items()},flush=True)
        del m,base,models,graphs,expected,ey,eu,x,s;gc.collect();torch.cuda.empty_cache()
    write(work/('pilot.json' if a.pilot else 'policy.json'),dict(finished=True,records=records,checkpoint_sha256=sha(p),
        rule='Independent synthetic 25Hz integer-ms event inputs; same-A nearest log input length; fastest valid complete forward by captured CUDA events; no accuracy/test inputs used for tuning.',gpu=os.environ.get('CUDA_VISIBLE_DEVICES')))

if __name__=='__main__':main()
