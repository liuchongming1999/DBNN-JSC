#!/usr/bin/env python3
"""Matched single-cell NEURON runs with explicit physical-core affinity.

Only the 31-manifest event rasters drive the cell. The dormant, zero-weight
NetStim connections made by the model constructor are removed in every
configuration, including the contemporaneous one-thread control.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

SCRIPT_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def stamp():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(median_ms=float(np.median(a)), mean_ms=float(a.mean()),
                std_ms=float(a.std(ddof=1)) if len(a) > 1 else 0.,
                runs_ms=a.tolist(), n=len(a))


def read_input(manifest_path, row):
    path = manifest_path.parent / row['input_file']
    if not path.exists():
        path = manifest_path.parent / 'inputs' / row['input_file']
    with np.load(path, allow_pickle=False) as f:
        ids, raster = f['ids'].astype(np.int32), f['x'].astype(np.uint8)
    assert hashlib.sha256(raster.tobytes()).hexdigest() == row['input_sha256']
    assert hashlib.sha256(ids.tobytes()).hexdigest() == row['ids_sha256']
    return ids, raster


def core_topology(cpus):
    rows = []
    for cpu in cpus:
        path = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        rows.append(dict(cpu=cpu, core=int((path/'core_id').read_text()),
                         socket=int((path/'physical_package_id').read_text()),
                         siblings=(path/'thread_siblings_list').read_text().strip()))
    assert len({(x['socket'], x['core']) for x in rows}) == len(rows)
    return rows


def build(args):
    sys.path.insert(0, str(args.root / 'scripts'))
    import sample_dbnn_data_repro as sample
    import neuron
    from neuron import h, coreneuron
    pc = h.ParallelContext()
    pc.nthread(args.threads)
    pc.thread_busywait(0)
    cfg = dict(root=str(args.root/'source'/'ShaiEtAl2015'),
               biophys_file='L5PCbiophys5.hoc', morphology_file='cell1.asc',
               num_synapses=500, duration_ms=16000., warmup_ms=500.,
               dt=args.dt, sample_interval_ms=1., input_rate_hz=50.,
               exc_weight=1.88e-4, inh_weight=8e-5,
               exc_tau1_ms=.5, exc_tau2_ms=5., inh_tau1_ms=1., inh_tau2_ms=10.,
               inh_reversal_mv=-80., v_init=-75., spike_threshold=-20., seed=20260425)
    started = time.perf_counter()
    state = sample.build_worker_network(cfg)
    # These objects supply no physical conductance (weights default to zero).
    # Remove their event overhead, and their zero-delay parallel connections.
    for ns in state['netstims']:
        ns.number = 0
    for nc in state['netcons']:
        nc.weight[0] = 0
    state['netcons'].clear()
    state['netstims'].clear()
    del ns, nc
    coreneuron.enable = False
    cv = h.CVode()
    cv.active(0)
    cv.cache_efficient(1)
    ncs = []
    for syn in state['synapses']:
        nc = h.NetCon(None, syn)
        nc.delay = 10.
        ncs.append(nc)
    spike_nc = h.NetCon(state['cell'].soma[0](.5)._ref_v, None, sec=state['cell'].soma[0])
    spike_nc.threshold = -20.
    # A target-less detector still contributes its delay to NEURON's parallel
    # mindelay calculation. Its default 1 ms is too small for dt=1 ms.
    # This metadata delay does not shift source-event recording or drive inputs.
    spike_nc.delay = 10.
    spike_vec = h.Vector()
    spike_nc.record(spike_vec)
    par = None
    if args.threads > 1:
        h.load_file('parcom.hoc')
        par = h.ParallelComputeTool()
        par.multisplit(1)
    h.dt = args.dt
    h.steps_per_ms = 1 / args.dt
    h.secondorder = 0
    pc.set_maxstep(10)
    pc.thread_stat()
    metadata = dict(neuron=neuron.__version__, neuron_build=h.nrnversion(6),
                    model=cfg, sections=sum(1 for _ in h.allsec()),
                    compartments=sum(int(s.nseg) for s in h.allsec()),
                    actual_threads=int(pc.nthread()), thread_busywait=0,
                    event_detector_metadata_delay_ms=10.,
                    affinity=sorted(os.sched_getaffinity(0)),
                    physical_cores=core_topology(sorted(os.sched_getaffinity(0))),
                    model_preparation_ms=(time.perf_counter()-started)*1000,
                    source_script_sha256=SCRIPT_SHA,
                    input_protocol='only manifest events; dormant zero-weight NetStims removed',
                    resources='shared server; disjoint physical cores, no exclusive reservation')
    if par is not None:
        metadata['multisplit'] = dict(pieces=float(par.lb.npiece_),
                                     estimated_load_balance=float(par.lb.thread_cxbal_))
    assert metadata['actual_threads'] == args.threads
    return h, pc, state, ncs, spike_nc, spike_vec, par, metadata


def run_case(args, runtime, row):
    h, pc, state, ncs, spike_nc, spike_vec, par, metadata = runtime
    name = f"nt{args.threads}_dt{args.dt:g}_{row['case_id']}"
    outj, outn = args.output_dir/(name+'.json'), args.output_dir/(name+'.npz')
    if outj.exists() and outn.exists():
        old = json.loads(outj.read_text())
        if old.get('finished') and old.get('source_script_sha256') == SCRIPT_SHA:
            return old
        raise RuntimeError(f'existing output from a different or incomplete run: {outj}')
    ids, raster = read_input(args.manifest, row)
    for nc in ncs:
        nc.weight[0] = 0.
    for site in ids:
        ncs[int(site)].weight[0] = 1.88e-4 if site < 250 else 8e-5
    # Record every fixed step exactly as in the expanded single-thread grid.
    vv = h.Vector().record(state['cell'].soma[0](.5)._ref_v)
    tt = h.Vector().record(h._ref_t)
    target = 500. + np.arange(raster.shape[1], dtype=np.float64)
    endpoint = float(target[-1])
    rep_records, outputs = [], []
    for rep in range(args.repeats):
        load_before = os.getloadavg()
        thread_before = [float(pc.thread_ctime(k)) for k in range(args.threads)]
        spike_vec.resize(0)
        started = time.perf_counter()
        h.dt = args.dt
        h.steps_per_ms = 1 / args.dt
        h.secondorder = 0
        h.finitialize(-75.)
        h.fcurrent()
        h.frecord_init()
        h.continuerun(500.)
        warm_end = time.perf_counter()
        for j, site in enumerate(ids):
            for k in np.flatnonzero(raster[j]):
                ncs[int(site)].event(float(500+k))
        schedule_end = time.perf_counter()
        solve_cpu_start = time.process_time()
        solve_start = time.perf_counter()
        pc.psolve(endpoint)
        solve_end = time.perf_counter()
        solve_cpu_end = time.process_time()
        times, values = np.asarray(tt.as_numpy()), np.asarray(vv.as_numpy())
        assert times.size == values.size and times.size > 1
        if times[-1] < endpoint-1e-5:
            values = np.append(values, float(state['cell'].soma[0](.5).v))
            times = np.append(times, endpoint)
        assert np.all(np.diff(times) > 0), 'recording timestamps not strictly increasing'
        assert times[0] <= target[0] and times[-1] >= endpoint-1e-5
        voltage = np.interp(target, times, values).astype(np.float32)
        assert np.isfinite(voltage).all()
        copy_end = time.perf_counter()
        sp = np.asarray(spike_vec.as_numpy())
        sp = sp[(sp >= 500-1e-6) & (sp <= endpoint+1e-6)] - 500
        thread_after = [float(pc.thread_ctime(k)) for k in range(args.threads)]
        rep_records.append(dict(warmup_ms=(warm_end-started)*1000,
                                schedule_ms=(schedule_end-warm_end)*1000,
                                solver_ms=(solve_end-solve_start)*1000,
                                output_ms=(copy_end-solve_end)*1000,
                                steady_ms=(copy_end-solve_start)*1000,
                                full_ms=(copy_end-started)*1000,
                                solver_cpu_ms=(solve_cpu_end-solve_cpu_start)*1000,
                                spike_times_ms=sp.tolist(), loadavg=load_before,
                                thread_integration_seconds=[a-b for a,b in zip(thread_after,thread_before)]))
        outputs.append(voltage.copy())
    voltage = outputs[-1]
    same_dt_npz = args.previous_results/f"neuron_dt{args.dt:g}_{row['case_id']}.npz"
    same_dt_json = same_dt_npz.with_suffix('.json')
    with np.load(same_dt_npz) as saved:
        baseline = saved['voltage'].astype(float).ravel()
    baseline_rec = json.loads(same_dt_json.read_text())
    assert baseline_rec['input_sha256'] == row['input_sha256']
    residual = voltage.astype(float) - baseline
    base_sp = np.asarray(baseline_rec['spike_times_ms'])
    event_match = base_sp.shape == sp.shape and bool(np.allclose(base_sp, sp, rtol=0, atol=1e-5))
    maxdiff = float(np.max(np.abs(residual)))
    validation = dict(against='original expanded NEURON single-thread at identical dt',
                      voltage_rmse_mv=float(np.sqrt(np.mean(residual**2))),
                      voltage_max_abs_mv=maxdiff,
                      event_times_equal_at_1e_5_ms=event_match,
                      previous_spike_times_ms=base_sp.tolist(),
                      repeat_voltage_max_abs_mv=float(np.max(np.abs(np.asarray(outputs)-outputs[0]))),
                      passed=maxdiff <= 1e-3 and event_match)
    timing_keys = ['warmup_ms','schedule_ms','solver_ms','output_ms','steady_ms','full_ms','solver_cpu_ms']
    record = dict(finished=True, created=stamp(), case_id=row['case_id'],
                  engine='neuron' if args.threads == 1 else 'multisplit',
                  threads=args.threads, dt_ms=args.dt, A=int(row['A']),
                  duration_ms=float(row['duration_ms']), site_mode=row['site_mode'],
                  rate_hz=float(row['rate_hz']), input_events=int(raster.sum()),
                  input_sha256=row['input_sha256'], ids_sha256=row['ids_sha256'],
                  spike_count=len(sp), spike_times_ms=sp.tolist(),
                  timing={k:stats([r[k] for r in rep_records]) for k in timing_keys},
                  repeats=rep_records, validation=validation, runtime=metadata,
                  source_script_sha256=SCRIPT_SHA, voltage_file=outn.name,
                  timer_scope='steady_ms=solver+recording+1ms-grid interpolation; initialization, 500ms warmup, scheduling and model build separate')
    np.savez_compressed(outn, voltage=voltage, ids=ids, spike_times_ms=sp)
    write_json(outj, record)
    if not validation['passed']:
        raise RuntimeError(f"output validation failed for {name}: {validation}")
    return record


def main():
    ap = argparse.ArgumentParser()
    for arg in ['root','manifest','output-dir','previous-results']:
        ap.add_argument('--'+arg, required=True, type=Path)
    ap.add_argument('--threads', type=int, choices=[1,2,4,8,16,32], required=True)
    ap.add_argument('--cpus', required=True)
    ap.add_argument('--dt', type=float, choices=[.25,.5,1.], required=True)
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--case-ids', nargs='*')
    args = ap.parse_args()
    for key in ['root','manifest','output_dir','previous_results']:
        setattr(args, key, getattr(args,key).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cpus = [int(x) for x in args.cpus.split(',')]
    assert len(cpus) == args.threads
    os.sched_setaffinity(0, cpus)
    core_topology(cpus)
    manifest = json.loads(args.manifest.read_text())
    rows = manifest['cases']
    if args.case_ids:
        rows = [r for cid in args.case_ids for r in rows if r['case_id'] == cid]
        assert len(rows) == len(args.case_ids)
    name = f'nt{args.threads}_dt{args.dt:g}'
    status_path = args.output_dir/f'status_{name}.json'
    start = stamp()
    records = []
    try:
        runtime = build(args)
        write_json(args.output_dir/f'runtime_{name}.json', runtime[-1])
        for index, row in enumerate(rows, 1):
            rec = run_case(args, runtime, row)
            records.append(dict(case_id=rec['case_id'], median_ms=rec['timing']['steady_ms']['median_ms'],
                                validation=rec['validation'], spike_count=rec['spike_count']))
            write_json(status_path, dict(started=start, updated=stamp(), finished=False,
                                         completed=index, total=len(rows), records=records))
            print(name, index, '/', len(rows), rec['case_id'],
                  'steady_ms', round(records[-1]['median_ms'], 3),
                  'max_voltage_diff', rec['validation']['voltage_max_abs_mv'], flush=True)
        write_json(status_path, dict(started=start, updated=stamp(), finished=True,
                                     completed=len(rows), total=len(rows), records=records))
    except Exception as error:
        write_json(status_path, dict(started=start, updated=stamp(), finished=False,
                                     completed=len(records), total=len(rows), records=records,
                                     error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
