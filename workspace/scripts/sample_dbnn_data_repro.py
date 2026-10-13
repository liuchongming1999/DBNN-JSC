#!/usr/bin/env python3
"""Reproduce the Shai et al. L5PC sampling protocol used by DBNN.

The generated NPZ stores binary input rasters and 1-ms soma-voltage targets.
The protocol follows the recovered experiment configuration: 500 synapses
(250 excitatory/250 inhibitory), 500-ms warm-up, 6-s recorded window,
NEURON dt=0.1 ms, 50-Hz Poisson input and fixed dendritic sites.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

WORKER_STATE = None


def parse_args():
    root = Path(__file__).resolve().parents[1] / "source" / "ShaiEtAl2015"
    default_output = Path(__file__).resolve().parents[1] / "data" / "shai_ei_500syn_6s_dataset.npz"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-samples", type=int, default=110)
    p.add_argument("--num-train", type=int, default=100)
    p.add_argument("--num-test", type=int, default=10)
    p.add_argument("--num-synapses", type=int, default=500)
    p.add_argument("--duration-ms", type=float, default=6000.0)
    p.add_argument("--warmup-ms", type=float, default=500.0)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--sample-interval-ms", type=float, default=1.0)
    p.add_argument("--input-rate-hz", type=float, default=50.0)
    p.add_argument("--exc-weight", type=float, default=1.88e-4)
    p.add_argument("--inh-weight", type=float, default=8.0e-5)
    p.add_argument("--exc-tau1-ms", type=float, default=0.5)
    p.add_argument("--exc-tau2-ms", type=float, default=5.0)
    p.add_argument("--inh-tau1-ms", type=float, default=1.0)
    p.add_argument("--inh-tau2-ms", type=float, default=10.0)
    p.add_argument("--inh-reversal-mv", type=float, default=-80.0)
    p.add_argument("--v-init", type=float, default=-75.0)
    p.add_argument("--spike-threshold", type=float, default=-20.0)
    p.add_argument("--seed", type=int, default=20260425)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--biophys-file", default="L5PCbiophys5.hoc")
    p.add_argument("--morphology-file", default="cell1.asc")
    p.add_argument("--output", type=Path, default=default_output)
    p.add_argument("--source-root", type=Path, default=root)
    return p.parse_args()


def ensure_mechanisms(root: Path):
    lib = root / "x86_64" / "libnrnmech.so"
    if lib.exists():
        return
    nrnivmodl = Path(sys.executable).with_name("nrnivmodl")
    if not nrnivmodl.exists():
        raise FileNotFoundError(f"Missing nrnivmodl at {nrnivmodl}")
    subprocess.run([str(nrnivmodl), "mod"], cwd=root, check=True)


def vec_to_np(vec):
    if hasattr(vec, "as_numpy"):
        return np.array(vec.as_numpy(), dtype=float, copy=True)
    return np.array(vec.to_python(), dtype=float)


def choose_sites_weighted(sections, count, rng):
    lengths = np.asarray([float(sec.L) for sec in sections], dtype=np.float64)
    probabilities = lengths / lengths.sum()
    indices = rng.choice(len(sections), size=count, replace=True, p=probabilities)
    return [(sections[i], float(rng.uniform(0.1, 0.9))) for i in indices]


def expected_event_count(total_ms, rate_hz):
    expected = total_ms * rate_hz / 1000.0
    return int(max(64, np.ceil(expected * 3.0)))


def build_worker_network(config):
    from neuron import h, load_mechanisms

    root = Path(config["root"])
    load_mechanisms(str(root))
    os.chdir(root / "simulationcode")
    h.load_file("stdrun.hoc")
    h.load_file("import3d.hoc")
    h.load_file(f"../models/{config['biophys_file']}")
    h.load_file("../models/L5PCtemplate.hoc")
    cell = h.L5PCtemplate(f"../morphologies/{config['morphology_file']}")
    dend_sections = list(cell.dend) + list(cell.apic)
    site_rng = np.random.default_rng(config["seed"])
    sites = choose_sites_weighted(dend_sections, config["num_synapses"], site_rng)
    excit_count = config["num_synapses"] // 2
    synapses, netstims, netcons = [], [], []
    channel_type, channel_section, channel_location = [], [], []
    interval_ms = 1000.0 / config["input_rate_hz"]
    event_count = expected_event_count(config["warmup_ms"] + config["duration_ms"], config["input_rate_hz"])
    for channel_idx, (section, location) in enumerate(sites):
        syn = h.Exp2Syn(section(location))
        is_exc = channel_idx < excit_count
        if is_exc:
            syn.e = 0.0
            syn.tau1 = config["exc_tau1_ms"]
            syn.tau2 = config["exc_tau2_ms"]
            channel_type.append("exc")
        else:
            syn.e = config["inh_reversal_mv"]
            syn.tau1 = config["inh_tau1_ms"]
            syn.tau2 = config["inh_tau2_ms"]
            channel_type.append("inh")
        netstim = h.NetStim()
        netstim.interval = interval_ms
        netstim.number = event_count
        netstim.noise = 1.0
        netstim.start = config["warmup_ms"]
        netcon = h.NetCon(netstim, syn)
        netcon.delay = 0.0
        synapses.append(syn)
        netstims.append(netstim)
        netcons.append(netcon)
        channel_section.append(section.name())
        channel_location.append(float(location))
    return {
        "h": h,
        "cell": cell,
        "synapses": synapses,
        "netstims": netstims,
        "netcons": netcons,
        "channel_type": np.asarray(channel_type),
        "channel_section": np.asarray(channel_section),
        "channel_location": np.asarray(channel_location, dtype=np.float32),
        "config": config,
    }


def init_worker(config):
    global WORKER_STATE
    WORKER_STATE = build_worker_network(config)
    WORKER_STATE["config"] = config


def run_single_sample(sample_index):
    state = WORKER_STATE
    h = state["h"]
    cell = state["cell"]
    config = state["config"]
    exc_scale = 1.0
    inh_scale = 1.0
    event_recorders = []
    for channel_idx, (channel_type, netstim, netcon) in enumerate(
        zip(state["channel_type"], state["netstims"], state["netcons"])
    ):
        netstim.noiseFromRandom123(config["seed"], sample_index + 1, channel_idx + 1)
        netstim.start = config["warmup_ms"]
        if channel_type == "exc":
            netcon.weight[0] = config["exc_weight"] * exc_scale
        else:
            netcon.weight[0] = config["inh_weight"] * inh_scale
        event_vec = h.Vector()
        netcon.record(event_vec)
        event_recorders.append(event_vec)

    voltage_vec = h.Vector().record(cell.soma[0](0.5)._ref_v)
    spike_vec = h.Vector()
    spike_nc = h.NetCon(cell.soma[0](0.5)._ref_v, None, sec=cell.soma[0])
    spike_nc.threshold = config["spike_threshold"]
    spike_nc.record(spike_vec)

    h.dt = config["dt"]
    h.steps_per_ms = int(round(1.0 / config["dt"]))
    h.finitialize(config["v_init"])
    h.fcurrent()
    h.frecord_init()
    h.continuerun(config["warmup_ms"] + config["duration_ms"])

    fine_voltage = vec_to_np(voltage_vec)
    fine_start = int(round(config["warmup_ms"] / config["dt"]))
    fine_stop = fine_start + int(round(config["duration_ms"] / config["dt"])) + 1
    stride = int(round(config["sample_interval_ms"] / config["dt"]))
    voltage_1ms = fine_voltage[fine_start:fine_stop:stride].astype(np.float32)
    spike_times = vec_to_np(spike_vec)
    mask = (spike_times >= config["warmup_ms"]) & (spike_times <= config["warmup_ms"] + config["duration_ms"])
    spike_times = np.round(spike_times[mask] - config["warmup_ms"], decimals=3)
    num_bins = int(round(config["duration_ms"] / config["sample_interval_ms"])) + 1
    ip = np.zeros((config["num_synapses"], num_bins), dtype=np.uint8)
    for channel_idx, event_vec in enumerate(event_recorders):
        event_times = vec_to_np(event_vec)
        mask = (event_times >= config["warmup_ms"]) & (event_times <= config["warmup_ms"] + config["duration_ms"])
        event_times = event_times[mask]
        if event_times.size == 0:
            continue
        bins = np.round((event_times - config["warmup_ms"]) / config["sample_interval_ms"]).astype(int)
        bins = bins[(bins >= 0) & (bins < num_bins)]
        if bins.size:
            ip[channel_idx, np.unique(bins)] = 1
    return {
        "sample_index": sample_index,
        "ip": ip,
        "op": voltage_1ms,
        "spike_time": spike_times.tolist(),
        "spike_count": int(spike_times.size),
        "exc_scale": exc_scale,
        "inh_scale": inh_scale,
    }


def main():
    args = parse_args()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.num_train + args.num_test != args.num_samples:
        raise ValueError("Expected num_train + num_test == num_samples.")
    if args.num_synapses % 2:
        raise ValueError("Expected an even number of synapses.")
    if not np.isclose(args.sample_interval_ms / args.dt, round(args.sample_interval_ms / args.dt)):
        raise ValueError("sample_interval_ms must be an integer multiple of dt.")

    source_root = args.source_root.resolve()
    ensure_mechanisms(source_root)
    config = {
        "root": str(source_root), "biophys_file": args.biophys_file, "morphology_file": args.morphology_file,
        "num_synapses": args.num_synapses, "duration_ms": args.duration_ms, "warmup_ms": args.warmup_ms,
        "dt": args.dt, "sample_interval_ms": args.sample_interval_ms, "input_rate_hz": args.input_rate_hz,
        "exc_weight": args.exc_weight, "inh_weight": args.inh_weight, "exc_tau1_ms": args.exc_tau1_ms,
        "exc_tau2_ms": args.exc_tau2_ms, "inh_tau1_ms": args.inh_tau1_ms, "inh_tau2_ms": args.inh_tau2_ms,
        "inh_reversal_mv": args.inh_reversal_mv, "v_init": args.v_init, "spike_threshold": args.spike_threshold,
        "seed": args.seed,
    }
    original_cwd = Path.cwd()
    init_worker(config)
    channel_type = WORKER_STATE["channel_type"].copy()
    channel_section = WORKER_STATE["channel_section"].copy()
    channel_location = WORKER_STATE["channel_location"].copy()
    os.chdir(original_cwd)

    num_bins = int(round(args.duration_ms / args.sample_interval_ms)) + 1
    ip_all = np.zeros((args.num_samples, args.num_synapses, num_bins), dtype=np.uint8)
    op_all = np.zeros((args.num_samples, num_bins), dtype=np.float32)
    spike_time_all = [None] * args.num_samples
    spike_count_all = np.zeros(args.num_samples, dtype=np.int32)
    exc_scale_all = np.zeros(args.num_samples, dtype=np.float32)
    inh_scale_all = np.zeros(args.num_samples, dtype=np.float32)
    sample_indices = list(range(args.num_samples))

    if args.num_workers <= 1:
        iterator = map(run_single_sample, sample_indices)
        for result in iterator:
            idx = result["sample_index"]
            ip_all[idx] = result["ip"]
            op_all[idx] = result["op"]
            spike_time_all[idx] = result["spike_time"]
            spike_count_all[idx] = result["spike_count"]
            exc_scale_all[idx] = result["exc_scale"]
            inh_scale_all[idx] = result["inh_scale"]
            print(f"sample {idx + 1:03d}/{args.num_samples}: spikes={result['spike_count']:>2d}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(args.num_workers, args.num_samples), initializer=init_worker, initargs=(config,)) as pool:
            for result in pool.imap_unordered(run_single_sample, sample_indices):
                idx = result["sample_index"]
                ip_all[idx] = result["ip"]
                op_all[idx] = result["op"]
                spike_time_all[idx] = result["spike_time"]
                spike_count_all[idx] = result["spike_count"]
                exc_scale_all[idx] = result["exc_scale"]
                inh_scale_all[idx] = result["inh_scale"]
                print(f"sample {idx + 1:03d}/{args.num_samples}: spikes={result['spike_count']:>2d}", flush=True)

    train_idx = np.arange(args.num_train, dtype=np.int32)
    test_idx = np.arange(args.num_train, args.num_train + args.num_test, dtype=np.int32)
    np.savez_compressed(
        args.output,
        train_ip=ip_all[train_idx], train_op=op_all[train_idx],
        train_spike_time=np.array([spike_time_all[i] for i in train_idx], dtype=object),
        train_spike_count=spike_count_all[train_idx],
        test_ip=ip_all[test_idx], test_op=op_all[test_idx],
        test_spike_time=np.array([spike_time_all[i] for i in test_idx], dtype=object),
        test_spike_count=spike_count_all[test_idx],
        channel_type=channel_type, channel_section=channel_section,
        channel_location=channel_location,
        train_exc_scale=exc_scale_all[train_idx], train_inh_scale=inh_scale_all[train_idx],
        test_exc_scale=exc_scale_all[test_idx], test_inh_scale=inh_scale_all[test_idx],
    )
    metadata = {
        "output": str(args.output), "num_samples": args.num_samples, "num_train": args.num_train,
        "num_test": args.num_test, "num_synapses": args.num_synapses,
        "num_exc_synapses": args.num_synapses // 2, "num_inh_synapses": args.num_synapses // 2,
        "duration_ms": args.duration_ms, "warmup_ms": args.warmup_ms, "dt": args.dt,
        "sample_interval_ms": args.sample_interval_ms, "input_rate_hz": args.input_rate_hz,
        "exc_weight": args.exc_weight, "inh_weight": args.inh_weight, "exc_tau1_ms": args.exc_tau1_ms,
        "exc_tau2_ms": args.exc_tau2_ms, "inh_tau1_ms": args.inh_tau1_ms, "inh_tau2_ms": args.inh_tau2_ms,
        "inh_reversal_mv": args.inh_reversal_mv, "v_init": args.v_init, "spike_threshold": args.spike_threshold,
        "seed": args.seed, "num_workers": args.num_workers,
        "dt_selection_note": "dt=0.1 ms protocol recovered from the original sampling script.",
        "train_spike_count_mean": float(np.mean(spike_count_all[train_idx])),
        "train_spike_count_std": float(np.std(spike_count_all[train_idx])),
        "train_spike_count_min": int(np.min(spike_count_all[train_idx])),
        "train_spike_count_max": int(np.max(spike_count_all[train_idx])),
        "test_spike_count_mean": float(np.mean(spike_count_all[test_idx])),
        "test_spike_count_std": float(np.std(spike_count_all[test_idx])),
        "test_spike_count_min": int(np.min(spike_count_all[test_idx])),
        "test_spike_count_max": int(np.max(spike_count_all[test_idx])),
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved dataset to {args.output}; train mean spikes={metadata['train_spike_count_mean']:.2f}, test mean spikes={metadata['test_spike_count_mean']:.2f}")


if __name__ == "__main__":
    main()
