"""Select observed simulator configurations under the frozen DBNN RMSE budget."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

H = Path(__file__).resolve().parent
R = H / 'matched_error_20260929'


def read(p):
    return json.loads(p.read_text())


def load_trace(record, folder=R):
    return np.load(folder / record['voltage_file'], allow_pickle=True)['voltage']


def stats(v):
    a = np.asarray(v, dtype=float)
    return dict(median_ms=float(np.median(a)), min_ms=float(a.min()), max_ms=float(a.max()),
                mean_ms=float(a.mean()), std_ms=float(a.std(ddof=1)) if a.size > 1 else 0.,
                n=int(a.size), runs_ms=a.tolist())


def build_summary():
    nr = read(R / 'neuron_dt0.1.json')
    assert nr['finished'] and nr['config']['samples'] == [0, 1, 2]
    reference = load_trace(nr)[0]
    assert reference.shape == (3, 6001)
    old_ref = np.load(H / 'results/fidelity_3x6s.npz', allow_pickle=True)['voltage'][0]
    ref_difference = float(np.max(np.abs(reference - old_ref)))
    # Native timestamp roundoff near a sharp spike yields micromV interpolation
    # differences relative to the earlier custom-interval recording.
    assert ref_difference < 1e-4, ref_difference
    ds = [read(R / f'dbnn_sample{s}.json') for s in range(3)]
    assert all(d['finished'] and d['implementation']['all_five_layers'] for d in ds)
    dbnn = np.concatenate([load_trace(d) for d in ds], axis=0)
    target = np.sqrt(np.mean((dbnn - reference) ** 2, axis=1))
    budget = float(np.sqrt(np.mean(target ** 2)))
    dbnn_runs = [d['transfer']['dense_gather_h2d_forward_d2h']['runs_ms'] for d in ds]
    dbnn_stats = stats(np.mean(dbnn_runs, axis=0))
    dbnn_stats['per_sample_median_ms'] = [float(np.median(v)) for v in dbnn_runs]
    # Verified directly in the frozen checkpoint on 2026-09-29.
    reset_threshold_mv = -57.5
    dbnn_events = [np.flatnonzero(v >= reset_threshold_mv).tolist() for v in dbnn]
    reference_spikes = nr['spikes'][0]
    rows, failed = [], []
    for p in sorted(R.glob('*_dt*.json')):
        r = read(p)
        if not r.get('finished'):
            failed.append(dict(file=p.name, error=r.get('error')))
            continue
        assert r['input_sha256'] == nr['input_sha256']
        assert r['active_ids_sha256'] == nr['active_ids_sha256']
        vs = load_trace(r)
        assert vs.shape[1:] == reference.shape and np.isfinite(vs).all()
        v = vs[0]
        errors = v - reference
        rmse = np.sqrt(np.mean(errors ** 2, axis=1))
        pooled = float(np.sqrt(np.mean(errors ** 2)))
        # Pair solve/output-copy within each trial and repeat before aggregating.
        per_trial = r['per_sample_timings_ms']
        solves = np.array([[z['solver_call'] for z in rep] for rep in per_trial])
        copies = np.array([[z['output_copy'] for z in rep] for rep in per_trial])
        trial = np.array([[z['trial_e2e'] for z in rep] for rep in per_trial])
        solve_stats = stats(solves.mean(axis=1))
        response_stats = stats((solves + copies).mean(axis=1))
        response_stats['per_sample_median_ms'] = np.median(solves + copies, axis=0).tolist()
        row = dict(mode=r['comparison_mode'], dt_ms=float(r['config']['dt']),
                   pooled_rmse_mv=pooled, per_sample_rmse_mv=rmse.tolist(),
                   max_abs_mv=float(np.max(np.abs(errors))),
                   within_pooled_budget=pooled <= budget,
                   within_every_sample_budget=bool(np.all(rmse <= target)),
                   error_ratio=pooled / budget, worst_sample_error_ratio=float(np.max(rmse / target)),
                   solver=solve_stats, response=response_stats, trial=stats(trial.mean(axis=1)),
                   spikes=r['spikes'][0], source_file=p.name,
                   sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                   created=r['created'], repeat_difference_mv=r['voltage_repeat_max_diff_mv'])
        rows.append(row)
    rows.sort(key=lambda r: (r['mode'], r['dt_ms']))
    selected, strict_selected, closest = {}, {}, {}
    for mode in sorted(set(r['mode'] for r in rows)):
        own = [r for r in rows if r['mode'] == mode]
        eligible = [r for r in own if r['within_pooled_budget']]
        strict = [r for r in own if r['within_every_sample_budget']]
        if eligible:
            selected[mode] = min(eligible, key=lambda r: r['response']['median_ms'])
        if strict:
            strict_selected[mode] = min(strict, key=lambda r: r['response']['median_ms'])
        closest[mode] = min(own, key=lambda r: abs(r['pooled_rmse_mv'] - budget))
    cpu_gpu_agreement = []
    for r in rows:
        if r['mode'] != 'core_gpu':
            continue
        matches = [q for q in rows if q['mode'] == 'core_cpu' and q['dt_ms'] == r['dt_ms']]
        if not matches:
            continue
        gpu_v = load_trace(read(R / r['source_file']))[0]
        cpu_v = load_trace(read(R / matches[0]['source_file']))[0]
        cpu_gpu_agreement.append(dict(dt_ms=r['dt_ms'], max_abs_mv=float(np.max(np.abs(gpu_v-cpu_v))),
                                      rmse_mv=float(np.sqrt(np.mean((gpu_v-cpu_v)**2)))))
    result = dict(created=time.strftime('%FT%T%z'), reference=dict(method='NEURON CPU 1T', dt_ms=.1,
                  samples=[0, 1, 2], steps=6001, output_interval_ms=1., duration_s=6., sites=500,
                  input_sha256=nr['input_sha256'], old_reference_max_difference_mv=ref_difference,
                  spikes=reference_spikes, source_file='neuron_dt0.1.json'),
                  dbnn=dict(pooled_rmse_mv=budget, per_sample_rmse_mv=target.tolist(),
                            response=dbnn_stats, samples=ds, reset_threshold_mv=reset_threshold_mv,
                            threshold_events_ms=dbnn_events),
                  criterion='Fastest measured solver+output time with pooled RMSE <= DBNN; also report per-input constraint',
                  rows=rows, selected=selected, strict_selected=strict_selected, closest=closest,
                  cpu_gpu_agreement=cpu_gpu_agreement, failed=failed, shared_resources=True,
                  all_modes_measured=all(k in selected for k in ['neuron', 'core_cpu', 'core_gpu']))
    (H / 'matched_error_summary.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    d = build_summary()
    print('DBNN', d['dbnn']['pooled_rmse_mv'], d['dbnn']['response']['median_ms'])
    for r in d['rows']:
        print(r['mode'], r['dt_ms'], 'RMSE', round(r['pooled_rmse_mv'], 5),
              [round(x, 5) for x in r['per_sample_rmse_mv']],
              'time', round(r['response']['median_ms'], 3),
              'pooled', r['within_pooled_budget'], 'all', r['within_every_sample_budget'])
    print('SELECTED', {k: v['dt_ms'] for k, v in d['selected'].items()})
    print('STRICT', {k: v['dt_ms'] for k, v in d['strict_selected'].items()})
    print('FAILED', d['failed'])
