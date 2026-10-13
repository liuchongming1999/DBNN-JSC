#!/usr/bin/env python3
"""Compare the new-shape fused path with the cuBLAS/reduction path."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--manifest",type=Path,required=True); ap.add_argument("--dbnn-dir",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--gpu",type=int,default=6)
    a=ap.parse_args(); a.root=a.root.resolve(); a.manifest=a.manifest.resolve(); a.output=a.output.resolve(); sys.path[:0]=[str(a.root/"source"),str(a.root/"benchmarks"/"sparse_20260928")]
    import torch
    from backends import Prepared,subset_state,capture
    from quad_optimized_backend import TiledPrepared
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    dev=torch.device("cuda"); ck=torch.load(a.root/"source/dbnn_checkpoint.pt",map_location="cpu",weights_only=False)["model_state"]; state_cpu={k:v.detach().clone() for k,v in ck.items()};state_cpu["bilinear.weight"][0].fill_diagonal_(0)
    m=json.loads(a.manifest.read_text()); rows=[]
    for row in m["cases"]:
        p=a.manifest.parent/row["input_file"]
        if not p.exists():p=a.manifest.parent/"inputs"/row["input_file"]
        d=np.load(p);ids=d["ids"].astype(np.int64);raw=d["x"].astype(np.float32);ii=torch.from_numpy(ids);signs=torch.where(ii<250,1.,-1.);packed=(torch.from_numpy(raw)*signs[:,None]).unsqueeze(0).contiguous().to(dev);s={k:v.to(dev) for k,v in state_cpu.items()};sp=subset_state(s,ii.to(dev));
        spec=dict(backend="triton",config=3,triangle=len(ids)>32,precision="ieee"); t=TiledPrepared(sp,packed,spec);t.block=512;tf=capture(t);b=Prepared(sp,packed,"triton","cublas");bf=capture(b)
        torch.cuda.synchronize();tv=tf().clone();bv=bf().clone();torch.cuda.synchronize();diff=(tv-bv).detach().float()
        rows.append(dict(case_id=row["case_id"],A=len(ids),T=raw.shape[1],
                         max_abs_mv=float(diff.abs().max().cpu()),rmse_mv=float(diff.square().mean().sqrt().cpu()),
                         finite=bool(torch.isfinite(tv).all()),
                         reset_count_fused=int(((tv[0,1:]>=-57.5)&(tv[0,:-1]<-57.5)).sum().cpu()),
                         reset_count_baseline=int(((bv[0,1:]>=-57.5)&(bv[0,:-1]<-57.5)).sum().cpu())))
        del s,sp,packed,t,tf,b,bf,tv,bv;torch.cuda.empty_cache();print(row["case_id"],rows[-1],flush=True)
    out=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),rows=rows,max_abs_mv=max(x["max_abs_mv"] for x in rows),rmse_mv_max=max(x["rmse_mv"] for x in rows),reset_count_mismatch=sum(x["reset_count_fused"]!=x["reset_count_baseline"] for x in rows),passed=all(x["finite"] and x["max_abs_mv"]<1e-3 and x["reset_count_fused"]==x["reset_count_baseline"] for x in rows))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2)+"\n")
if __name__=="__main__":main()
