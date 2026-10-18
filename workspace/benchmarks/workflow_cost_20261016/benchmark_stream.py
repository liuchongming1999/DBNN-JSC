"""Observed full-panel service calls with a bounded layout cache, fixed weights."""
from pathlib import Path
from collections import OrderedDict
import argparse,json,time,os,sys,gc,hashlib
import numpy as np
import torch
from benchmark_cost import ROOT,WORK,CP,SHA,setup,input_data,read,write,sha

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['gpu','cpu8'],required=True);p.add_argument('--cpus',required=True)
    a=p.parse_args();setup(a);gpu=a.method=='gpu';nt=1 if gpu else 8
    torch.set_num_threads(nt);torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from gpu_backend import ResolutionModel,Graph
    from backends import subset_state
    from cpu_backend import CPUModel
    from benchmark_cpu import methods
    assert sha(CP)==SHA
    state=torch.load(CP,map_location='cuda' if gpu else 'cpu',weights_only=False)['model_state']
    locked=read(WORK/'locked_backends.json')['methods'][a.method]
    rows=read(WORK/'manifest.json')['cases'];out=WORK/'stream'/a.method;out.mkdir(parents=True,exist_ok=True)
    def close(obj):
        if not gpu:obj[0].close()
    for round_index in range(3):
        cache=OrderedDict();records=[]
        for i,row in enumerate(rows):
            ids,raw=input_data(row);tag=locked[row['case_id']]
            # Output oracle loading is deliberately outside service-call timing.
            with np.load(ROOT/'benchmarks/random_init_main379_20261012/predictions'/f"{row['case_id']}.npz") as f:
                expected=f['voltage'].copy();events=f['predicted_events_ms'].copy()
            key=(row['ids_sha256'],raw.shape[1],tag)
            if gpu:torch.cuda.synchronize()
            t=time.perf_counter();hit=key in cache
            signed=np.ascontiguousarray(raw*np.where(ids[:,None]<250,1,-1),dtype=np.int8)
            if not hit:
                if gpu:
                    x=torch.from_numpy(signed)[None].cuda()
                    st=subset_state(state,torch.as_tensor(ids.astype(np.int64),device='cuda'))
                    mode,block=tag.rsplit('_',1);m=ResolutionModel(st,x,1.,mode,int(block));graph=Graph(m)
                else:
                    x=np.zeros((500,raw.shape[1]),np.float32);m=CPUModel(state,ids.astype(np.int64),raw.shape[1]);graph=None
                cache[key]=(m,graph,x)
                if len(cache)>16:close(cache.popitem(last=False)[1])
            else:cache.move_to_end(key)
            m,graph,x=cache[key]
            if gpu:
                x.copy_(torch.from_numpy(signed)[None]);graph();v=m.out[0].cpu().numpy().copy();threshold=m.th
            else:
                x.fill(0);x[ids]=signed;v=m.run(x,nt,**methods()[tag]).copy();threshold=m.threshold
            e=np.flatnonzero(v>=threshold).astype(float);elapsed=(time.perf_counter()-t)*1000
            gap=float(np.max(np.abs(v-expected)));eq=bool(np.array_equal(e,events));assert gap<.001 and eq,(row['case_id'],gap,eq)
            records.append(dict(case_id=row['case_id'],duration_s=row['duration_s'],A=row['A'],
                input_sha256=row['input_sha256'],ids_sha256=row['ids_sha256'],layout_cache_hit=hit,
                elapsed_ms=elapsed,validation=dict(max_abs_mv=gap,events_equal=eq)))
            if (i+1)%50==0:print(a.method,round_index,i+1,flush=True)
        for obj in cache.values():close(obj)
        del cache;gc.collect()
        write(out/f'round{round_index}.json',dict(finished=True,cache_capacity=16,
              order='Frozen manifest order; empty layout cache at start of each round',records=records,
              checkpoint_sha256=sha(CP),sum_ms=sum(x['elapsed_ms'] for x in records)))
    write(out/'complete.json',dict(finished=True,rounds=3,inputs=379))
if __name__=='__main__':main()
