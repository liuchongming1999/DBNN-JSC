#!/usr/bin/env python3
"""Measure the Triton scan+reset inference path at batch sizes 1..8."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'));sys.path.insert(0,str(ROOT/'scripts'))
from DBNN import DBNN_F,DBNN_R
from dbnn_triton_inference import DBNNTritonInference

def bench(fn, sync, reps=8):
    for _ in range(2): fn()
    sync(); ts=[]
    for _ in range(reps):
        sync(); t=time.perf_counter(); fn(); sync(); ts.append(time.perf_counter()-t)
    return {"median_sec":float(np.median(ts)),"mean_sec":float(np.mean(ts)),"std_sec":float(np.std(ts)),"runs_sec":[float(x) for x in ts]}

def main():
    device=torch.device("cuda")
    d=np.load(ROOT/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True)
    signs=np.where(d['channel_type'].astype(str)=='exc',1.,-1.).astype(np.float32)
    x=torch.from_numpy(d['train_ip'][:8].astype(np.float32)*signs[None,:,None]).to(device)
    ckpt=torch.load(ROOT/'source/dbnn_checkpoint.pt',map_location='cpu',weights_only=False)
    out={"device":str(device),"torch":torch.__version__,"results":{}}
    tri=DBNNTritonInference(ROOT/'source/dbnn_checkpoint.pt',device)
    for B in [1,2,4,8]:
        xb=x[:B]
        # Like-for-like reference model outputs.
        models={"DBNN_F":DBNN_F(500,6001,device),"DBNN_R":DBNN_R(500,6001,device,scan_chunk_size=256)}
        for name,m in models.items():
            m.load_state_dict(ckpt['model_state'],strict=True);m.eval()
            fn=lambda m=m: m(xb)
            with torch.inference_mode(): ref=fn(); torch.cuda.synchronize()
            
            def ref_fn(m=m, xb=xb):
                with torch.inference_mode(): return m(xb)
            rec=bench(ref_fn,lambda:torch.cuda.synchronize())
            # Re-run error outside timing.
            with torch.inference_mode(): err=m(xb)-ref
            rec.update({"rmse_vs_reference_mv":0.0,"max_abs_vs_reference_mv":0.0})
            out['results'].setdefault(str(B),{})[name]=rec
        with torch.inference_mode(): ref=models['DBNN_R'](xb); triout=tri(xb); torch.cuda.synchronize()
        e=triout-ref
        rec=bench(lambda:tri(xb),lambda:torch.cuda.synchronize())
        rec.update({"rmse_vs_reference_mv":float(torch.sqrt((e*e).mean())),"max_abs_vs_reference_mv":float(e.abs().max())})
        out['results'].setdefault(str(B),{})['DBNN_Triton']=rec
    path=ROOT/'results'/'triton_batch_benchmark_20260928.json';path.write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
