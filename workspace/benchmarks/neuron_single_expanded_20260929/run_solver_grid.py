#!/usr/bin/env python3
"""Run one fixed-step NEURON or CoreNEURON CPU grid on selected cases."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np


def stamp(): return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def stat(x):
    a=np.asarray(x,dtype=float)
    return dict(median_ms=float(np.median(a)),mean_ms=float(a.mean()),std_ms=float(a.std(ddof=1)) if len(a)>1 else 0.,runs_ms=a.tolist())


def one_case(h, coreneuron, pc, spike_nc, spike_vec, state, ncs, row, input_dir, out_dir, engine, dt, repeats):
    stem=f"{engine}_dt{dt:g}_{row['case_id']}"
    oj=out_dir/f"{stem}.json"; on=out_dir/f"{stem}.npz"
    if oj.exists() and on.exists() and json.loads(oj.read_text()).get("finished"):
        return json.loads(oj.read_text())
    input_path=input_dir/row["input_file"]
    if not input_path.exists(): input_path=input_dir/"inputs"/row["input_file"]
    d=np.load(input_path,allow_pickle=False); ids=d["ids"].astype(np.int32); x=d["x"].astype(np.uint8)
    if hashlib.sha256(x.tobytes()).hexdigest()!=row["input_sha256"]: raise ValueError("input hash")
    for nc in ncs: nc.weight[0]=0.0
    for site in ids: ncs[int(site)].weight[0]=1.88e-4 if int(site)<250 else 8e-5
    target=500.0+np.arange(x.shape[1],dtype=np.float64); endpoint=float(target[-1])
    vv=h.Vector().record(state["cell"].soma[0](.5)._ref_v); tt=h.Vector().record(h._ref_t)
    rep_records=[]
    for rep in range(repeats):
        spike_vec.resize(0)
        t0=time.perf_counter(); h.dt=dt; h.steps_per_ms=1.0/dt; h.secondorder=0
        coreneuron.enable=False
        h.finitialize(-75.0); h.fcurrent(); h.frecord_init()
        h.continuerun(500.0)
        warm_end=time.perf_counter()
        for j,site in enumerate(ids):
            nc=ncs[int(site)]
            for k in np.flatnonzero(x[j]): nc.event(float(500.0+k))
        schedule_end=time.perf_counter()
        coreneuron.enable=(engine=="core"); coreneuron.gpu=False; coreneuron.cell_permute=0; coreneuron.warp_balance=1; coreneuron.verbose=0
        solve_start=time.perf_counter(); pc.psolve(endpoint); solve_end=time.perf_counter()
        times=np.asarray(tt.as_numpy(),dtype=np.float64); vals=np.asarray(vv.as_numpy(),dtype=np.float64)
        if times.size==0 or times[-1]<endpoint-1e-5:
            vals=np.append(vals,float(state["cell"].soma[0](.5).v));times=np.append(times,endpoint)
        out=np.interp(target,times,vals).astype(np.float32)
        copy_end=time.perf_counter()
        sp=np.asarray(spike_vec.as_numpy(),dtype=np.float64); sp=sp[(sp>=500-1e-6)&(sp<=endpoint+1e-6)]-500.0
        rep_records.append(dict(warmup_ms=(warm_end-t0)*1000.0,schedule_ms=(schedule_end-warm_end)*1000.0,
                                solver_ms=(solve_end-solve_start)*1000.0,output_ms=(copy_end-solve_end)*1000.0,
                                steady_ms=(copy_end-solve_start)*1000.0,full_ms=(copy_end-t0)*1000.0,
                                spike_times_ms=sp.tolist()))
    np.savez_compressed(on,voltage=out,ids=ids,spike_times=np.asarray(rep_records[-1]["spike_times_ms"]))
    rec=dict(finished=True,created=stamp(),case_id=row["case_id"],engine=engine,dt_ms=float(dt),
             duration_ms=float(row["duration_ms"]),A=int(row["A"]),site_mode=row["site_mode"],rate_hz=float(row["rate_hz"]),
             spike_count=int(len(rep_records[-1]["spike_times_ms"])),spike_times_ms=rep_records[-1]["spike_times_ms"],
             input_sha256=row["input_sha256"],ids_sha256=row["ids_sha256"],input_events=int(x.sum()),voltage_file=on.name,
             timing={k:stat([r[k] for r in rep_records]) for k in ["warmup_ms","schedule_ms","solver_ms","output_ms","steady_ms","full_ms"]},
             timer_scope="steady_ms=solver+common-grid interpolation; full_ms also includes 500-ms initialization/warmup and event scheduling; model build separate")
    oj.write_text(json.dumps(rec,indent=2)+"\n"); return rec


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--manifest",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True); ap.add_argument("--engine",choices=["neuron","core"],required=True)
    ap.add_argument("--dt",type=float,required=True); ap.add_argument("--repeats",type=int,default=2)
    args=ap.parse_args(); args.root=args.root.resolve(); args.manifest=args.manifest.resolve(); args.output_dir=args.output_dir.resolve(); args.output_dir.mkdir(parents=True,exist_ok=True)
    root=args.root; m=json.loads(args.manifest.read_text()); sys.path.insert(0,str(root/"scripts")); import sample_dbnn_data_repro as sample
    from neuron import h,coreneuron
    cfg=dict(root=str(root/"source"/"ShaiEtAl2015"),biophys_file="L5PCbiophys5.hoc",morphology_file="cell1.asc",num_synapses=500,duration_ms=16000.,warmup_ms=500.,dt=args.dt,sample_interval_ms=1.,input_rate_hz=50.,exc_weight=1.88e-4,inh_weight=8e-5,exc_tau1_ms=.5,exc_tau2_ms=5.,inh_tau1_ms=1.,inh_tau2_ms=10.,inh_reversal_mv=-80.,v_init=-75.,spike_threshold=-20.,seed=20260425)
    state=sample.build_worker_network(cfg); coreneuron.enable=False; pc=h.ParallelContext();pc.nthread(1);pc.thread_busywait(0);pc.set_maxstep(10)
    spike_nc=h.NetCon(state["cell"].soma[0](.5)._ref_v,None,sec=state["cell"].soma[0]); spike_nc.threshold=-20.0
    spike_vec=h.Vector(); spike_nc.record(spike_vec)
    if args.engine=="core": pc.set_gid2node(0,0); pc.cell(0,spike_nc)
    ncs=[]
    for syn in state["synapses"]:
        nc=h.NetCon(None,syn);nc.delay=10.;ncs.append(nc)
    rows=[]; status=args.output_dir/f"status_{args.engine}_dt{args.dt:g}.json"
    for i,row in enumerate(m["cases"],1):
        try:
            rec=one_case(h,coreneuron,pc,spike_nc,spike_vec,state,ncs,row,args.manifest.parent,args.output_dir,args.engine,args.dt,args.repeats); rows.append({k:rec.get(k) for k in ("case_id","finished","spike_count","timing")})
            print(i,len(m["cases"]),row["case_id"],"spikes",rec.get("spike_count"),"steady_ms",round(rec["timing"]["steady_ms"]["median_ms"],2),flush=True)
        except Exception as e:
            rec=dict(finished=False,case_id=row["case_id"],error=repr(e)); (args.output_dir/f"{args.engine}_dt{args.dt:g}_{row['case_id']}.json").write_text(json.dumps(rec,indent=2)+"\n"); rows.append(rec); print(i,row["case_id"],"ERROR",repr(e),flush=True)
        status.write_text(json.dumps(dict(updated=stamp(),completed=i,total=len(m["cases"]),records=rows),indent=2)+"\n")
    (args.output_dir/f"complete_{args.engine}_dt{args.dt:g}.json").write_text(json.dumps(dict(finished=True,records=rows),indent=2)+"\n")

if __name__=="__main__": main()
