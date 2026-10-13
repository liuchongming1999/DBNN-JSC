#!/usr/bin/env python3
"""Aggregate expanded matched-RMSE and latency results."""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path
import numpy as np


def load_json(p): return json.loads(Path(p).read_text())
def rms(x): return float(np.sqrt(np.mean(np.asarray(x,dtype=float)**2)))
def stat(x):
    a=np.asarray(x,dtype=float)
    return dict(median_ms=float(np.median(a)),mean_ms=float(a.mean()),std_ms=float(a.std(ddof=1)) if len(a)>1 else 0.,n=int(len(a)))
def crossings(v, threshold=-20.0):
    v=np.asarray(v,dtype=float); above=v>=threshold
    return int(above[0])+int(np.sum(above[1:] & ~above[:-1])) if v.size else 0
def crossing_times(v, threshold=-20.0):
    v=np.asarray(v,dtype=float); above=v>=threshold
    return (np.flatnonzero(above[1:] & ~above[:-1])+1).astype(float).tolist()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--manifest",type=Path,required=True); ap.add_argument("--solver-dir",type=Path,required=True); ap.add_argument("--dbnn-dir",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args(); m=load_json(args.manifest); cases=[]
    for row in m["cases"]:
        rf=Path(row["reference_file"]); rd=load_json(rf); rv=np.load(rf.with_suffix(".npz"))["voltage"].astype(float)
        item=dict(case_id=row["case_id"],duration_s=row["duration_s"],duration_ms=row["duration_ms"],A=row["A"],site_mode=row["site_mode"],rate_hz=row["rate_hz"],reference_spikes=rd["spike_times_ms"],spike_class=rd["spike_count"],reference_file=str(rf),reference_voltage_file=rd["voltage_file"])
        item["reference_voltage"]=rv
        dbf=args.dbnn_dir/f"dbnn_{row['case_id']}.json"; db=np.load(dbf.with_suffix(".npz"))["voltage"].astype(float).reshape(-1); dj=load_json(dbf)
        dbnn_threshold=-57.5
        item["dbnn_voltage"]=db; item["dbnn"]={"rmse_mv":rms(db-rv),"max_abs_mv":float(np.max(np.abs(db-rv))),"spike_count":crossings(db,dbnn_threshold),"spike_times_ms":crossing_times(db,dbnn_threshold),"threshold_mv":dbnn_threshold,"timing":dj["timing"]}
        cases.append(item)
    solver_rows=[]; modes=[]
    for p in sorted(args.solver_dir.glob("*.json")):
        if p.name.startswith("status_") or p.name.startswith("complete_"): continue
        sj=load_json(p)
        if not sj.get("finished"): continue
        cid=sj["case_id"]; match=next(c for c in cases if c["case_id"]==cid)
        sv=np.load(p.with_suffix(".npz"))["voltage"].astype(float).reshape(-1); ref=match["reference_voltage"]
        r=dict(case_id=cid,engine=sj["engine"],dt_ms=sj["dt_ms"],duration_s=match["duration_s"],A=match["A"],site_mode=match["site_mode"],rate_hz=match["rate_hz"],spike_class=match["spike_class"],reference_spikes=match["reference_spikes"],voltage=sv,rmse_mv=rms(sv-ref),max_abs_mv=float(np.max(np.abs(sv-ref))),spike_count=int(np.sum(sv>=-20.0)),timing=sj["timing"],source_file=p.name)
        solver_rows.append(r); modes.append((sj["engine"],sj["dt_ms"]))
    dbnn_rmse=np.array([r["dbnn"]["rmse_mv"] for r in cases])
    dbnn_micro_num=sum(np.sum((r["dbnn_voltage"]-r["reference_voltage"])**2) for r in cases); dbnn_den=sum(r["reference_voltage"].size for r in cases)
    budget=dict(micro_rmse_mv=float(np.sqrt(dbnn_micro_num/dbnn_den)),macro_rmse_mv=rms(dbnn_rmse),mean_case_rmse_mv=float(dbnn_rmse.mean()),case_count=len(cases),point_count=dbnn_den)
    budget["spike_count_exact_fraction"]=float(np.mean([r["dbnn"]["spike_count"]==len(r["reference_spikes"]) for r in cases]))
    budget["spike_count_mae"]=float(np.mean([abs(r["dbnn"]["spike_count"]-len(r["reference_spikes"])) for r in cases]))
    budget["by_spike_class"]={}
    for cat in [0,1,2,3]:
        cc=[r for r in cases if r["spike_class"]==cat]
        if cc: budget["by_spike_class"][str(cat)]=dict(n=len(cc),macro_rmse_mv=rms([r["dbnn"]["rmse_mv"] for r in cc]),spike_count_exact_fraction=float(np.mean([r["dbnn"]["spike_count"]==cat for r in cc])))
    aggregates=[]
    for engine,dt in sorted(set(modes)):
        rows=[r for r in solver_rows if r["engine"]==engine and r["dt_ms"]==dt]
        num=sum(np.sum((r["voltage"]-next(c for c in cases if c["case_id"]==r["case_id"])["reference_voltage"])**2) for r in rows); den=sum(next(c for c in cases if c["case_id"]==r["case_id"])["reference_voltage"].size for r in rows)
        er=np.array([r["rmse_mv"] for r in rows]);
        steady=np.array([r["timing"]["steady_ms"]["median_ms"] for r in rows]); full=np.array([r["timing"]["full_ms"]["median_ms"] for r in rows]);
        db_dense=np.array([next(c for c in cases if c["case_id"]==r["case_id"])["dbnn"]["timing"]["dense_gather_e2e"]["median_ms"] for r in rows])
        ref_counts=np.array([len(r["reference_spikes"]) for r in rows]); solver_counts=np.array([r["spike_count"] for r in rows]);
        agg=dict(engine=engine,dt_ms=dt,case_count=len(rows),micro_rmse_mv=float(np.sqrt(num/den)),macro_rmse_mv=rms(er),mean_case_rmse_mv=float(er.mean()),max_case_rmse_mv=float(er.max()),steady_time_ms=stat(steady),full_time_ms=stat(full),dbnn_dense_e2e_ms=stat(db_dense),total_steady_ms=float(steady.sum()),total_dbnn_dense_e2e_ms=float(db_dense.sum()),total_steady_speedup=float(steady.sum()/db_dense.sum()),steady_speedup_vs_dbnn_median=float(np.median(steady/db_dense)),steady_speedup_geomean=float(np.exp(np.mean(np.log(steady/db_dense)))),spike_count_exact_fraction=float(np.mean(solver_counts==ref_counts)),spike_count_mae=float(np.mean(np.abs(solver_counts-ref_counts))),within_dbnn_macro=bool(rms(er)<=budget["macro_rmse_mv"]),within_dbnn_micro=bool(np.sqrt(num/den)<=budget["micro_rmse_mv"]),within_dbnn_macro_and_spike_counts=bool(rms(er)<=budget["macro_rmse_mv"] and np.all(solver_counts==ref_counts)))
        cats={}
        for cat in [0,1,2,3]:
            cr=[r for r in rows if r["spike_class"]==cat]
            if cr: cats[str(cat)]=dict(n=len(cr),macro_rmse_mv=rms([r["rmse_mv"] for r in cr]),mean_solver_spikes=float(np.mean([r["spike_count"] for r in cr])),mean_reference_spikes=float(np.mean([len(r["reference_spikes"]) for r in cr])))
        agg["by_spike_class"]=cats; aggregates.append(agg)
    # Match NEURON and CoreNEURON outputs at the same dt before comparing them
    # to DBNN; this isolates the backend from the discretization effect.
    agreement=[]
    for dt in sorted({r["dt_ms"] for r in solver_rows}):
        nr={r["case_id"]:r for r in solver_rows if r["engine"]=="neuron" and r["dt_ms"]==dt}; cr={r["case_id"]:r for r in solver_rows if r["engine"]=="core" and r["dt_ms"]==dt};
        vals=[rms(nr[k]["voltage"]-cr[k]["voltage"]) for k in nr.keys() if k in cr]; mx=[float(np.max(np.abs(nr[k]["voltage"]-cr[k]["voltage"]))) for k in nr.keys() if k in cr]
        if vals: agreement.append(dict(dt_ms=dt,cases=len(vals),rmse_mv_max=max(vals),max_abs_mv_max=max(mx),rmse_mv_mean=float(np.mean(vals))))
    # JSON-friendly case summary omits voltage arrays but keeps per-case errors.
    public_cases=[]
    for c in cases:
        public_cases.append({k:c[k] for k in ["case_id","duration_s","A","site_mode","rate_hz","spike_class","reference_spikes","reference_file","reference_voltage_file"]}|{"dbnn_rmse_mv":c["dbnn"]["rmse_mv"],"dbnn_max_abs_mv":c["dbnn"]["max_abs_mv"],"dbnn_spike_count":c["dbnn"]["spike_count"],"dbnn_spike_times_ms":c["dbnn"]["spike_times_ms"],"dbnn_threshold_mv":c["dbnn"]["threshold_mv"],"dbnn_timing":c["dbnn"]["timing"]})
    for r in solver_rows:
        public_cases.append({"case_id":r["case_id"],"engine":r["engine"],"dt_ms":r["dt_ms"],"rmse_mv":r["rmse_mv"],"max_abs_mv":r["max_abs_mv"],"spike_class":r["spike_class"],"solver_spike_count":r["spike_count"],"timing":r["timing"],"source_file":r["source_file"]})
    out=dict(created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),protocol=m["protocol"],budget=budget,aggregates=aggregates,backend_agreement=agreement,cases=public_cases,notes=["Reference is NEURON dt=0.1 ms; all outputs are on a common 1 ms grid.","Primary set excludes trajectories with more than three reference spikes; they are preserved in stress_cases.json.","Macro RMSE gives each complete trajectory equal weight; micro RMSE pools all voltage samples.","NEURON/CoreNEURON timings are steady solver plus common-grid output interpolation; full timing is reported separately."])
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps(dict(budget=budget,aggregates=aggregates,agreement=agreement),indent=2))

if __name__=="__main__": main()
