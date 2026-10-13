#!/usr/bin/env python3
"""Run NEURON dt=0.1 ms reference trajectories for the expanded manifest."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def run_case(h, state, active_ncs, row, input_dir, out_dir, warmup_ms=500.0):
    stem = row["case_id"]
    out_json = out_dir / f"{stem}_reference.json"
    out_npz = out_dir / f"{stem}_reference.npz"
    if out_json.exists() and json.loads(out_json.read_text()).get("finished") and out_npz.exists():
        return json.loads(out_json.read_text())
    d = np.load(input_dir / row["input_file"], allow_pickle=False)
    ids = d["ids"].astype(np.int32)
    x = d["x"].astype(np.uint8)
    if hashlib.sha256(x.tobytes()).hexdigest() != row["input_sha256"]:
        raise ValueError(f"input hash mismatch for {stem}")
    duration_ms = float(row["duration_ms"])
    dt = 0.1
    target = warmup_ms + np.arange(x.shape[1], dtype=np.float64)
    # Disable the 500 NetStim/NetCon objects from build_worker_network. The
    # explicit event objects below replay the manifest exactly.
    for ns in state["netstims"]:
        ns.number = 0
    for nc in state["netcons"]:
        nc.weight[0] = 0.0
    for nc in active_ncs:
        nc.weight[0] = 0.0
    for j, site in enumerate(ids):
        active_ncs[int(site)].weight[0] = 1.88e-4 if int(site) < 250 else 8.0e-5
    vv = h.Vector().record(state["cell"].soma[0](0.5)._ref_v)
    tt = h.Vector().record(h._ref_t)
    spike_nc = h.NetCon(state["cell"].soma[0](0.5)._ref_v, None, sec=state["cell"].soma[0])
    spike_nc.threshold = -20.0
    spikes = h.Vector(); spike_nc.record(spikes)
    start = time.perf_counter()
    h.dt = dt; h.steps_per_ms = int(round(1.0 / dt)); h.secondorder = 0
    h.finitialize(-75.0); h.fcurrent(); h.frecord_init()
    h.continuerun(warmup_ms)
    schedule_start = time.perf_counter()
    for j, site in enumerate(ids):
        nc = active_ncs[int(site)]
        for k in np.flatnonzero(x[j]):
            nc.event(float(warmup_ms + k))
    schedule_ms = (time.perf_counter() - schedule_start) * 1000.0
    h.continuerun(warmup_ms + duration_ms)
    solve_ms = (time.perf_counter() - schedule_start) * 1000.0 - schedule_ms
    times = np.asarray(tt.as_numpy(), dtype=np.float64)
    vals = np.asarray(vv.as_numpy(), dtype=np.float64)
    if times.size == 0 or times[-1] < target[-1] - 1e-6:
        vals = np.append(vals, float(state["cell"].soma[0](0.5).v)); times = np.append(times, target[-1])
    voltage = np.interp(target, times, vals).astype(np.float32)
    spike_times = np.asarray(spikes.as_numpy(), dtype=np.float64)
    spike_times = spike_times[(spike_times >= warmup_ms - 1e-6) & (spike_times <= warmup_ms + duration_ms + 1e-6)] - warmup_ms
    voltage_file = out_npz.name
    np.savez_compressed(out_npz, voltage=voltage, ids=ids, spike_times=spike_times)
    rec = dict(finished=True, created=stamp(), case_id=stem, engine="neuron", dt_ms=dt,
               output_interval_ms=1.0, warmup_ms=warmup_ms, duration_ms=duration_ms,
               A=int(row["A"]), site_mode=row["site_mode"], rate_hz=float(row["rate_hz"]),
               input_sha256=row["input_sha256"], ids_sha256=row["ids_sha256"],
               input_events=int(x.sum()), spike_times_ms=spike_times.tolist(),
               spike_count=int(spike_times.size), voltage_file=voltage_file,
               schedule_ms=float(schedule_ms), solve_ms=float(solve_ms),
               wall_ms=float((time.perf_counter() - start) * 1000.0),
               recording="native dt=0.1 ms then linear interpolation to 1 ms grid")
    out_json.write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--cpu", type=int, default=74)
    args = ap.parse_args()
    # build_worker_network changes cwd into the NEURON model directory; resolve
    # every campaign path before importing/building the model.
    args.root = args.root.resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = args.root; manifest = json.loads(args.manifest.read_text())
    sys.path.insert(0, str(root / "scripts"))
    import sample_dbnn_data_repro as sample
    from neuron import h, coreneuron
    cfg = dict(root=str(root / "source" / "ShaiEtAl2015"), biophys_file="L5PCbiophys5.hoc",
               morphology_file="cell1.asc", num_synapses=500, duration_ms=16000.0,
               warmup_ms=500.0, dt=0.1, sample_interval_ms=1.0, input_rate_hz=50.0,
               exc_weight=1.88e-4, inh_weight=8e-5, exc_tau1_ms=.5, exc_tau2_ms=5.,
               inh_tau1_ms=1., inh_tau2_ms=10., inh_reversal_mv=-80., v_init=-75.,
               spike_threshold=-20., seed=20260425)
    state = sample.build_worker_network(cfg)
    coreneuron.enable = False
    pc = h.ParallelContext(); pc.nthread(1); pc.thread_busywait(0); pc.set_maxstep(10)
    # One explicit NetCon per stored synapse makes all cases share the same
    # physical sites while allowing arbitrary integer-ms event rasters.
    active_ncs = []
    for site, syn in enumerate(state["synapses"]):
        nc = h.NetCon(None, syn); nc.delay = 10.0; active_ncs.append(nc)
    rows = []; status = args.output_dir / "reference_status.json"
    for i, row in enumerate(manifest["cases"], 1):
        try:
            rec = run_case(h, state, active_ncs, row, args.manifest.parent, args.output_dir)
            rows.append({k: rec.get(k) for k in ("case_id", "finished", "spike_count", "wall_ms", "error")})
            print(i, len(manifest["cases"]), row["case_id"], "spikes", rec.get("spike_count"), "wall_ms", round(rec.get("wall_ms", 0), 1), flush=True)
        except Exception as e:
            rec = dict(finished=False, case_id=row["case_id"], error=repr(e))
            (args.output_dir / f"{row['case_id']}_reference.json").write_text(json.dumps(rec, indent=2) + "\n")
            rows.append(rec); print(i, row["case_id"], "ERROR", repr(e), flush=True)
        status.write_text(json.dumps(dict(updated=stamp(), completed=i, total=len(manifest["cases"]), records=rows), indent=2) + "\n")
    (args.output_dir / "reference_complete.json").write_text(json.dumps(dict(finished=True, records=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
