#!/usr/bin/env python3
"""Matched NEURON/DBNN backend benchmark on one fixed reproduced sample."""
from __future__ import annotations
import argparse, json, platform, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(ROOT / "scripts"))
from DBNN import DBNN_F, DBNN_R
try:
    from triton_backend import dual_exp_scan
except Exception:
    dual_exp_scan = None


def args():
    p=argparse.ArgumentParser()
    p.add_argument("--sample-index",type=int,default=0)
    p.add_argument("--repeats",type=int,default=5)
    p.add_argument("--gpu-device",default="cuda:0")
    p.add_argument("--output",type=Path,default=ROOT/"results"/"backend_benchmark_20260928.json")
    p.add_argument("--skip-neuron",action="store_true")
    return p.parse_args()


def load_input(sample_index):
    d=np.load(ROOT/"data/shai_ei_500syn_6s_dataset.npz",allow_pickle=True)
    signs=np.where(d["channel_type"].astype(str)=="exc",1.,-1.).astype(np.float32)
    x=d["train_ip"][sample_index:sample_index+1].astype(np.float32)*signs[None,:,None]
    y=d["train_op"][sample_index:sample_index+1].astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(y), d["channel_type"].astype(str)


def load_model(cls, state, device, **kw):
    m=cls(500,6001,device,**kw); m.load_state_dict(state,strict=True); m.eval(); return m


def bench(fn, sync=None, repeats=5, warmup=2):
    for _ in range(warmup): fn()
    if sync: sync()
    times=[]
    for _ in range(repeats):
        if sync: sync()
        t=time.perf_counter(); fn()
        if sync: sync()
        times.append(time.perf_counter()-t)
    return {"median_sec":float(np.median(times)),"mean_sec":float(np.mean(times)),"std_sec":float(np.std(times)),"runs_sec":[float(v) for v in times]}


def main():
    a=args(); a.output=a.output.resolve(); a.output.parent.mkdir(parents=True,exist_ok=True)
    x,y,ch=load_input(a.sample_index)
    ckpt=torch.load(ROOT/"source/dbnn_checkpoint.pt",map_location="cpu",weights_only=False)
    state=ckpt["model_state"]
    results={"project":str(ROOT),"sample_index":a.sample_index,"input_shape":list(x.shape),"torch":torch.__version__,"platform":platform.platform(),"results":{}}
    for threads in [1,8,32]:
        torch.set_num_threads(threads)
        for name,cls,kw in [("DBNN_F_cpu",DBNN_F,{}),("DBNN_R_cpu",DBNN_R,{"scan_chunk_size":256})]:
            m=load_model(cls,state,torch.device("cpu"),**kw)
            def fn(m=m):
                with torch.inference_mode(): return m(x)
            out=fn(); e=out-y
            rec=bench(fn,repeats=a.repeats)
            rec.update({"threads":threads,"rmse_vs_neuron_mv":float(torch.sqrt((e*e).mean())),"max_abs_vs_neuron_mv":float(e.abs().max())})
            results["results"][f"{name}_threads{threads}"]=rec
    if torch.cuda.is_available():
        dev=torch.device(a.gpu_device)
        xg=x.to(dev); yg=y.to(dev)
        for name,cls,kw in [("DBNN_F_gpu",DBNN_F,{}),("DBNN_R_gpu",DBNN_R,{"scan_chunk_size":256})]:
            m=load_model(cls,state,dev,**kw)
            def fn(m=m):
                with torch.inference_mode():
                    return m(xg)
            with torch.inference_mode(): out=fn(); torch.cuda.synchronize(dev)
            e=out-yg
            rec=bench(fn,sync=lambda:torch.cuda.synchronize(dev),repeats=a.repeats)
            rec.update({"device":str(dev),"rmse_vs_neuron_mv":float(torch.sqrt((e*e).mean())),"max_abs_vs_neuron_mv":float(e.abs().max())})
            results["results"][name]=rec
        if dual_exp_scan is not None:
            m=load_model(DBNN_R,state,dev,scan_chunk_size=256)
            W=m.bilinear._effective_weight().squeeze(0)
            def triton_fn():
                with torch.inference_mode():
                    yy=dual_exp_scan(xg,m.tau_rise,m.tau_decay,m.omega); yb=yy.permute(0,2,1)
                    drive=((yb@W)*yb).sum(-1)+yb.sum(-1)+m.bilinear.bias
                    # Use the reference reset path for a like-for-like voltage output.
                    from DBNN import _apply_reset_closed_loop
                    return _apply_reset_closed_loop(drive,m.v_th,m.reset_amp,m.tau_reset,m.surr_beta)
            with torch.inference_mode(): out=triton_fn(); torch.cuda.synchronize(dev)
            e=out-yg; rec=bench(triton_fn,sync=lambda:torch.cuda.synchronize(dev),repeats=a.repeats)
            rec.update({"device":str(dev),"rmse_vs_neuron_mv":float(torch.sqrt((e*e).mean())),"max_abs_vs_neuron_mv":float(e.abs().max())})
            results["results"]["DBNN_R_triton_gpu"]=rec
    if not a.skip_neuron:
        import sample_dbnn_data_repro as s
        project=ROOT; cfg={"root":str((project/"source/ShaiEtAl2015").resolve()),"biophys_file":"L5PCbiophys5.hoc","morphology_file":"cell1.asc","num_synapses":500,"duration_ms":6000.,"warmup_ms":500.,"dt":.1,"sample_interval_ms":1.,"input_rate_hz":50.,"exc_weight":1.88e-4,"inh_weight":8e-5,"exc_tau1_ms":.5,"exc_tau2_ms":5.,"inh_tau1_ms":1.,"inh_tau2_ms":10.,"inh_reversal_mv":-80.,"v_init":-75.,"spike_threshold":-20.,"seed":20260425}
        s.init_worker(cfg)
        def nfn(): return s.run_single_sample(a.sample_index)
        # One timed run; the sampled input/target identity is also verified.
        t=time.perf_counter(); rr=nfn(); elapsed=time.perf_counter()-t
        stored=np.load(ROOT/"data/shai_ei_500syn_6s_dataset.npz",allow_pickle=True)
        results["results"]["NEURON_cpu"]={"median_sec":elapsed,"mean_sec":elapsed,"std_sec":0.,"runs_sec":[elapsed],"spike_count":rr["spike_count"],"input_exact":bool(np.array_equal(stored["train_ip"][a.sample_index],rr["ip"])),"voltage_exact_rmse_mv":float(np.sqrt(np.mean((stored["train_op"][a.sample_index]-rr["op"])**2)))}
    a.output.write_text(json.dumps(results,indent=2),encoding="utf-8")
    print(json.dumps(results,indent=2))

if __name__=="__main__": main()
