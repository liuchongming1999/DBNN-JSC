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
    input_dt = float(row['input_interval_ms'])
    target = warmup_ms + np.arange(round(duration_ms / 0.5) + 1, dtype=np.float64) * 0.5
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
            for _ in range(int(x[j, k])):
                nc.event(float(warmup_ms + k * input_dt))
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
    np.savez_compressed(out_npz, voltage=voltage, ids=ids, spike_times=spike_times, time_ms=target-warmup_ms)
    rec = dict(finished=True, created=stamp(), case_id=stem, engine="neuron", dt_ms=dt,
               output_interval_ms=0.5, input_interval_ms=input_dt, warmup_ms=warmup_ms, duration_ms=duration_ms,
               A=int(row["A"]), site_mode=row["site_mode"], rate_hz=float(row["rate_hz"]),
               input_sha256=row["input_sha256"], ids_sha256=row["ids_sha256"],
               input_events=int(x.sum()), spike_times_ms=spike_times.tolist(),
               spike_count=int(spike_times.size), voltage_file=voltage_file,
               schedule_ms=float(schedule_ms), solve_ms=float(solve_ms),
               wall_ms=float((time.perf_counter() - start) * 1000.0),
               recording="native dt=0.1 ms then linear interpolation to 0.5 ms grid")
    out_json.write_text(json.dumps(rec, indent=2) + "\n")
    return rec
