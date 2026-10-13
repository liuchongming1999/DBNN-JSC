#!/usr/bin/env python3
"""Generate deterministic active-site / Poisson input cases for the campaign."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np


def pick_sites(types, loc, A, mode, seed):
    types = np.asarray(types).astype(str)
    loc = np.asarray(loc, dtype=float)
    if A == len(types):
        return np.arange(len(types), dtype=np.int32)
    rng = np.random.default_rng(seed)
    chosen = []
    per = (A // 2, A - A // 2)
    for kind, count in zip(("exc", "inh"), per):
        pool = np.flatnonzero(types == kind)
        if mode == "random":
            cand = pool
        elif mode == "proximal":
            cand = pool[np.argsort(loc[pool])[: max(count * 2, count)]]
        elif mode == "distal":
            cand = pool[np.argsort(loc[pool])[-max(count * 2, count):]]
        elif mode == "stratified":
            order = pool[np.argsort(loc[pool])]
            # Spread selected sites over the complete stored location range.
            q = np.linspace(0, len(order) - 1, count).round().astype(int)
            chosen.extend(order[q].tolist())
            continue
        else:
            raise ValueError(mode)
        chosen.extend(rng.choice(cand, size=count, replace=False).tolist())
    return np.sort(np.asarray(chosen, dtype=np.int32))


def make_raster(ids, duration_ms, rate_hz, seed):
    rng = np.random.default_rng(seed)
    # One binary DBNN input bin per physical millisecond, including endpoints.
    p = float(rate_hz) / 1000.0
    return (rng.random((len(ids), int(duration_ms) + 1)) < p).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--replicates", type=int, default=2)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    d = np.load(args.dataset, allow_pickle=True)
    types, loc = d["channel_type"], d["channel_location"]
    # This first pool is deliberately broad. Reference screening later selects
    # a balanced subset with 0/1/2/3 spikes; no case is discarded silently.
    durations_s = [1, 2, 4, 8, 16]
    active_counts = [64, 128, 256, 500]
    site_modes = ["random", "proximal", "distal", "stratified"]
    rates = [10.0, 25.0, 50.0]
    rows = []
    case_index = 0
    for duration_s in durations_s:
        # Avoid an unnecessary Cartesian explosion: all durations and A values
        # are covered, while the site/rate panel is replicated deterministically.
        for A in active_counts:
            modes = ["all"] if A == 500 else ["random", "stratified"]
            rate_panel = rates if (A == 500 or duration_s <= 4) else [10.0, 25.0]
            for mode in modes:
                for rate_hz in rate_panel:
                    for rep in range(args.replicates):
                        seed = args.seed + 100000 * duration_s + 1000 * A + 100 * int(rate_hz) + 10 * rep + len(rows)
                        ids = pick_sites(types, loc, A, mode if mode != "all" else "random", seed)
                        x = make_raster(ids, duration_s * 1000, rate_hz, seed + 17)
                        stem = f"case_{case_index:04d}"
                        np.savez_compressed(args.output_dir / f"{stem}.npz", ids=ids, x=x)
                        row = dict(case_id=stem, duration_s=duration_s, duration_ms=duration_s * 1000,
                                   A=A, site_mode=mode, rate_hz=rate_hz, replicate=rep,
                                   seed=seed, input_file=f"{stem}.npz",
                                   input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                                   ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                                   input_events=int(x.sum()))
                        rows.append(row); case_index += 1
    manifest = dict(created="2026-09-29", dataset=str(args.dataset.resolve()),
                    protocol=dict(warmup_ms=500.0, input_grid_ms=1.0, reference_dt_ms=0.1,
                                  spike_threshold_mv=-20.0, durations_s=durations_s,
                                  active_counts=active_counts, rates_hz=rates,
                                  replicates=args.replicates,
                                  note="Input rasters are independent Poisson draws on the 1 ms grid; active IDs are exact stored L5PC sites."),
                    cases=rows)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "by_duration": {str(t): sum(r["duration_s"] == t for r in rows) for t in durations_s}}, indent=2))


if __name__ == "__main__":
    main()
