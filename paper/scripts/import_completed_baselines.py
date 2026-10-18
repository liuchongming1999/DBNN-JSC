"""Import the fixed-budget ELM and paired, quiet-device timing without fitting.

Run after the reference-refinement importer. All original input memberships,
DBNN/LSTM/TCN predictions and numerical-solver results are preserved.
"""
from pathlib import Path
import copy, hashlib, json
import numpy as np
from event_metrics import match_events
from update_refined_reference import summary

R = Path(__file__).resolve().parents[1]
E = R / 'evidence'
S = E / 'baseline_final_20261017'


def read(p):
    return json.loads(p.read_text())


def main():
    if not S.exists():
        return
    train = read(S / 'training.json')
    ev = read(S / 'elm_accuracy.json')
    moments = read(S / 'elm_moments.json')
    gpu = read(S / 'paired_gpu.json')
    audit = read(S / 'timing_audit.json')
    assert train['finished'] and train['default_budget_completed']
    assert train['updates'] == train['total_updates'] == 350000
    assert gpu['finished'] and gpu['full_results'] and gpu['checkpoints_unchanged']
    assert len(gpu['records']) == len(ev['records']) == len(moments['records']) == 379
    assert audit['all_samples_free_of_other_compute_processes']
    assert all(v['passed'] for row in gpu['records'] for v in row['parity'].values())
    assert gpu['protocol']['repeats'] == 7
    assert train['checkpoint_sha256'] == ev['corrected_checkpoint_sha256']
    d = read(E / 'main379_results.json')
    before = copy.deepcopy(d)
    frozen = {r['case_id']: r for r in d['models']['gpu']['records']}
    new = {r['case_id']: r for r in moments['records']}
    timed = {r['case_id']: r for r in gpu['records']}
    assert set(new) == set(timed) == set(frozen)
    for cid, row in frozen.items():
        assert timed[cid]['input_sha256'] == row['input_sha256']
        assert new[cid]['reference_events_ms'] == row['reference_events_ms']
    m = d['models']['elm']
    for row in m['records']:
        n = new[row['case_id']]
        row.update(copy.deepcopy(n))
        row['events'] = match_events(row['reference_events_ms'], row['predicted_events_ms'])
    m['accuracy'] = summary(m['records'])
    m['groups'] = {axis: {name: summary([r for r in m['records'] if str(r[axis]) == name])
                          for name in groups} for axis, groups in m['groups'].items()}
    assert abs(m['accuracy']['pooled_rmse_mv'] - ev['summary']['elm_corrected']['rmse_mv']) < 1e-12
    mapping = dict(gpu='dbnn_frozen', tcn='tcn_whole', elm='elm_corrected', lstm='lstm_unchanged')
    for key, remote in mapping.items():
        records = []
        for row in d['models'][key]['records']:
            rec = timed[row['case_id']]
            t = rec['timing'][remote]
            assert len(t['runs_ms']) == 7 and np.median(t['runs_ms']) == t['median_ms']
            records.append(dict(case_id=row['case_id'], duration_s=row['duration_s'], A=row['A'],
                input_sha256=row['input_sha256'], checkpoint_sha256=gpu['protocol']['models'][remote]['sha256'],
                wall=dict(median_ms=t['median_ms'], runs_ms=t['runs_ms'],
                          q1_ms=float(np.quantile(t['runs_ms'], .25)), q3_ms=float(np.quantile(t['runs_ms'], .75))),
                backend='CUDA Graph', validation=dict(rec['parity'][remote],
                    max_abs_mv=rec['parity'][remote]['max_voltage_gap_mv'])))
        total = sum(r['wall']['median_ms'] for r in records)
        assert np.isclose(total, gpu['summary'][remote]['wall_ms'], rtol=1e-12)
        d['models'][key]['timing'] = dict(records=records, wall_ms=total, sum_median_ms=total,
            finished=True, source='baseline_final_20261017/paired_gpu.json', repeats=7)
        if key != 'elm':
            assert d['models'][key]['records'] == before['models'][key]['records']
            assert d['models'][key]['accuracy'] == before['models'][key]['accuracy']
    for duration, models in d['durations'].items():
        for key in mapping:
            vals = [r['wall']['median_ms'] for r in d['models'][key]['timing']['records'] if r['duration_s'] == int(duration)]
            models[key] = dict(cases=len(vals), median_ms=float(np.median(vals)),
                q1_ms=float(np.quantile(vals, .25)), q3_ms=float(np.quantile(vals, .75)), sum_ms=sum(vals))
    d['protocol'].update(learned_repeats=7, solver_repeats=3,
        gpu_timing_lane='GPU0, randomized method order per input; device compute-process monitoring archived',
        learned_inference='TCN whole trajectory; ELM/LSTM original overlap; all CUDA Graph forwards include calibration/sigmoid')
    d['baseline_revision'] = dict(source='baseline_final_20261017', elm_checkpoint_sha256=train['checkpoint_sha256'],
        elm_updates=350000, common_fit=117, common_validation=32, no_test_based_selection=True,
        tcn_whole_trajectory=True, dbnn_lstm_tcn_weights_and_thresholds_unchanged=True,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in S.glob('*.json')})
    (E / 'main379_results.json').write_text(json.dumps(d, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(updated=True, elm=m['accuracy']['pooled_rmse_mv'],
        gpu_times_s={k:d['models'][k]['timing']['wall_ms']/1000 for k in mapping}), indent=2))


if __name__ == '__main__':
    main()
