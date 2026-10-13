#!/usr/bin/env python3
"""Add event-class controls based on the original held-out raster.

The original 50-Hz test rasters contain naturally occurring passive, one-,
two- and three-spike trajectories. Reusing their event times makes the
category labels auditable rather than trying to force a category by looking at
the DBNN prediction.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from generate_manifest import pick_sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.dataset, allow_pickle=True)
    types, loc = d["channel_type"], d["channel_location"]
    base = d["test_ip"].astype(np.uint8)
    rows=[]; idx=0
    def add(name, duration_s, A, mode, sample, rate_hz=50.0, thin=1.0, zero=False):
        nonlocal idx
        seed=20261000+idx
        ids = pick_sites(types, loc, A, mode if mode != "all" else "random", seed)
        if zero:
            x=np.zeros((len(ids), duration_s*1000+1), dtype=np.uint8)
        else:
            src=base[sample][:, :min(base.shape[-1], duration_s*1000+1)]
            src=src[ids]
            if thin < 1.0:
                keep=np.random.default_rng(seed+1).random(src.shape)<thin
                src=src*keep.astype(np.uint8)
            x=np.zeros((len(ids), duration_s*1000+1), dtype=np.uint8)
            x[:, :src.shape[1]]=src
        stem=f"target_{idx:03d}"; np.savez_compressed(args.output_dir/f"{stem}.npz",ids=ids,x=x)
        rows.append(dict(case_id=stem,duration_s=duration_s,duration_ms=duration_s*1000,A=A,
                         site_mode=mode,rate_hz=rate_hz,replicate=0,seed=seed,
                         source_kind="original_test_raster" if not zero else "zero_input",
                         source_sample=None if zero else int(sample),thinning=thin,
                         input_file=f"{stem}.npz",input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                         ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),input_events=int(x.sum())))
        idx+=1
    # In the original held-out set, sample 0 has one spike by 6 s, sample 1
    # has two, and sample 2 has three. Truncation/zero-padding preserves the
    # physical event raster and yields auditable 4/8/16-s examples.
    for T in [4,8,16]:
        for s in [0,1,2]: add(f"base{s}_T{T}",T,500,"all",s)
    # Site-position sensitivity on the same two-spike source raster.
    for A,mode in [(64,"random"),(128,"stratified"),(256,"proximal"),(256,"distal")]:
        add(f"base1_A{A}_{mode}",8,A,mode,1)
    # Lower event strength controls, retaining the source timing as a known
    # 50-Hz raster with independent thinning.
    for T in [8,16]: add(f"thin_base2_T{T}",T,500,"all",2,rate_hz=25.,thin=.5)
    # Explicit passive controls across time scales and active-site counts.
    for T,A in [(1,64),(8,256),(16,500)]: add(f"passive_T{T}_A{A}",T,A,"random",0,zero=True)
    out=dict(created="2026-09-29",protocol=dict(reference_dt_ms=.1,output_interval_ms=1.,
                note="Targeted event classes use original held-out rasters or zero input; labels are assigned from NEURON reference after simulation."),cases=rows)
    (args.output_dir/"manifest.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"cases":len(rows),"output":str(args.output_dir)},indent=2))

if __name__=="__main__": main()
