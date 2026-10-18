"""Diagnose frozen manuscript predictions without training, inference or retuning.

The chronological one-to-one matcher uses a uniform one-nanosecond boundary
margin. A separate dynamic program verifies match cardinalities.
All bootstrap comparisons use the same sampled complete input trajectories.
"""
from pathlib import Path
import base64
import csv
import hashlib
import html
import json
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PAPER = Path(__file__).resolve().parents[1]
SOURCE = PAPER / 'evidence/main379_results.json'
CHECKPOINT = Path('/Users/lcmsjtu/Documents/ChatGPT/博士毕业论文/实验脚本/dbnn_random_init_main379_20261012/results/DBNN_random_init_1ms.pt')
OUT = PAPER / 'qa/refined_spike_diagnostics'
TOLS = [1, 2, 5, 10, 20]
KEYS = ['gpu', 'elm', 'lstm', 'tcn', 'neuron8', 'neuron8_dt0.25', 'neuron8_dt0.1', 'reduce']
NAMES = dict(gpu='DBNN', elm='Branch-ELM', lstm='LSTM', tcn='TCN',
             neuron8='NEURON / CoreNEURON', reduce='Neuron_Reduce')
COLORS = dict(gpu='#008779', elm='#475668', lstm='#218FAF', tcn='#CB513D',
              neuron8='#D68A28', reduce='#97713D')
NAMES.update({'neuron8':'NEURON / CoreNEURON, 0.5 ms','neuron8_dt0.25':'NEURON / CoreNEURON, 0.25 ms','neuron8_dt0.1':'NEURON / CoreNEURON, 0.1 ms'})
COLORS.update({'neuron8_dt0.25':'#B16D20','neuron8_dt0.1':'#7C4B16'})
BOOTSTRAPS = 10000
SEED = 20261016
BOUNDARY_EPS_MS = 1e-6


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def match_events(ref, pred, tol=5., boundary_eps_ms=BOUNDARY_EPS_MS):
    ref = sorted(map(float, ref))
    pred = sorted(map(float, pred))
    i = j = 0
    dt = []
    pairs = []
    tol = float(tol) + float(boundary_eps_ms)
    while i < len(ref) and j < len(pred):
        if pred[j] < ref[i] - tol:
            j += 1
        elif ref[i] < pred[j] - tol:
            i += 1
        else:
            dt.append(pred[j] - ref[i])
            pairs.append((i, j))
            i += 1
            j += 1
    return dict(tp=len(dt), fp=len(pred)-len(dt), fn=len(ref)-len(dt),
                matched_errors_ms=dt, pairs=pairs)


def independent_dp_count(ref, pred, tol):
    """Maximum-cardinality matching with chronological order, separately coded."""
    ref, pred = sorted(ref), sorted(pred)
    tol = float(tol) + BOUNDARY_EPS_MS
    dp = np.zeros((len(ref)+1, len(pred)+1), dtype=np.int64)
    for i, a in enumerate(ref, 1):
        for j, b in enumerate(pred, 1):
            dp[i, j] = max(dp[i-1, j], dp[i, j-1])
            # Use inclusive endpoints and the same tiny numerical margin.
            if not (b < a-tol or a < b-tol):
                dp[i, j] = max(dp[i, j], dp[i-1, j-1]+1)
    return int(dp[-1, -1])


def score_counts(tp, fp, fn):
    n = 2*tp+fp+fn
    return dict(tp=int(tp), fp=int(fp), fn=int(fn),
                precision=float(tp/(tp+fp)) if tp+fp else None,
                recall=float(tp/(tp+fn)) if tp+fn else None,
                f1=float(2*tp/n) if n else None)


def quantiles(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                q25=float(np.quantile(a, .25)), q75=float(np.quantile(a, .75)),
                p90=float(np.quantile(a, .9)), p95=float(np.quantile(a, .95)),
                max=float(a.max()), min=float(a.min()))


def zero_stats(records):
    duration = sum(r['duration_s'] for r in records)
    false_events = sum(len(r['predicted_events_ms']) for r in records)
    positive = sum(bool(r['predicted_events_ms']) for r in records)
    return dict(cases=len(records), duration_s=duration,
                trajectories_with_predicted_events=positive,
                trajectory_fraction_with_predicted_events=positive/len(records),
                extra_events=false_events, extra_events_per_minute=false_events/(duration/60))


def fmt(value, digits=3):
    return '—' if value is None else f'{value:.{digits}f}'


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows, classes=''):
    return '<div class="scroll"><table class="'+classes+'"><thead><tr>'+''.join(
        '<th>'+html.escape(str(s))+'</th>' for s in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+str(s)+'</td>' for s in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def figure(path, caption):
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    assets = OUT / 'figures'
    assets.mkdir(exist_ok=True)
    protected = [SOURCE, PAPER/'main.pdf', *sorted((PAPER/'sections').glob('*.tex'))]
    if CHECKPOINT.exists():protected.append(CHECKPOINT)
    before = {str(p): sha(p) for p in protected}
    data = json.loads(SOURCE.read_text())
    if CHECKPOINT.exists():assert sha(CHECKPOINT) == data['checkpoint_sha256']
    original_records = data['models']['gpu']['records']
    ids = [r['case_id'] for r in original_records]
    assert len(ids) == len(set(ids)) == 379
    assert sum(len(r['reference_events_ms']) for r in original_records) == data['reference_events'] == 680
    assert sum(r['duration_s'] for r in original_records) == 2497
    frozen_reference = {r['case_id']: r for r in original_records}
    stochastic = np.array([i for i, r in enumerate(original_records) if r['family'] != 'zero'])
    controls = np.array([i for i, r in enumerate(original_records) if r['family'] == 'zero'])
    assert (len(stochastic), len(controls)) == (374, 5)
    # The bootstrap resamples complete inputs, identically for all methods/tolerances.
    weights = np.random.default_rng(SEED).multinomial(len(stochastic),
        np.full(len(stochastic), 1/len(stochastic)), size=BOOTSTRAPS)
    counts = np.zeros((len(KEYS), len(TOLS), 379, 3), dtype=np.int64)
    results = {}
    csv_tolerance, csv_cases = [], []
    all_errors = {}
    dp_checks = 0
    endpoint_counts = {}
    for mi, key in enumerate(KEYS):
        model = data['models'][key]
        records = model['records']
        assert [r['case_id'] for r in records] == ids
        errors = {t: [] for t in TOLS}
        for ci, r in enumerate(records):
            original = frozen_reference[r['case_id']]
            assert r['input_sha256'] == original['input_sha256']
            assert r['ids_sha256'] == original['ids_sha256']
            assert r['duration_s'] == original['duration_s']
            assert r['reference_events_ms'] == original['reference_events_ms']
            assert r['reference_spikes'] == len(r['reference_events_ms'])
            ref = r['reference_events_ms']
            pred = r['predicted_events_ms']
            assert all(math.isfinite(x) for x in ref+pred)
            assert ref == sorted(ref) and pred == sorted(pred)
            v = r['voltage']
            assert np.isclose(math.sqrt(v['sse']/v['n']), v['rmse_mv'], rtol=1e-12, atol=1e-12)
            for ti, tol in enumerate(TOLS):
                matched = match_events(ref, pred, tol)
                assert matched['tp'] == independent_dp_count(ref, pred, tol)
                dp_checks += 1
                counts[mi, ti, ci] = [matched[n] for n in ('tp', 'fp', 'fn')]
                errors[tol].extend(matched['matched_errors_ms'])
                if tol == 5:
                    assert all(matched[n] == r['events'][n] for n in ('tp', 'fp', 'fn'))
            five = match_events(ref, pred, 5)
            csv_cases.append(dict(method=key, case_id=r['case_id'], family=r['family'],
                duration_s=r['duration_s'], active_sites=r['A'], input_sha256=r['input_sha256'],
                reference_events=len(ref), predicted_events=len(pred),
                signed_count_error=len(pred)-len(ref), absolute_count_error=abs(len(pred)-len(ref)),
                tp_at_5ms=five['tp'], fp_at_5ms=five['fp'], fn_at_5ms=five['fn'],
                masked_samples=v['n'], masked_sse_mv2=v['sse'], masked_rmse_mv=v['rmse_mv']))
        total = counts[mi].sum(axis=1)
        # Check sensitivity to floating-point endpoints separately, keeping the
        # original exact convention as the primary analysis.
        endpoint_counts[key] = {}
        for ti, tol in enumerate(TOLS):
            legacy = sum(match_events(r['reference_events_ms'], r['predicted_events_ms'], tol, boundary_eps_ms=0.)['tp'] for r in records)
            endpoint_counts[key][str(tol)] = dict(original_tp=legacy,
                tp_with_one_nanosecond_slack=int(total[ti, 0]), extra_matches=int(total[ti, 0])-legacy)
        sensitivity = {}
        for ti, tol in enumerate(TOLS):
            tp, fp, fn = total[ti]
            scores = score_counts(tp, fp, fn)
            bootstrap_counts = weights @ counts[mi, ti, stochastic] + counts[mi, ti, controls].sum(axis=0)
            f1 = 2*bootstrap_counts[:, 0]/(2*bootstrap_counts[:, 0]+bootstrap_counts[:, 1]+bootstrap_counts[:, 2])
            scores['f1_ci95_paired_trajectory_bootstrap'] = np.quantile(f1, [.025, .975]).tolist()
            scores['signed_matched_time_error_ms'] = quantiles(errors[tol]) if errors[tol] else None
            scores['median_abs_matched_time_error_ms'] = float(np.median(np.abs(errors[tol]))) if errors[tol] else None
            sensitivity[str(tol)] = scores
            csv_tolerance.append(dict(method=key, name=NAMES[key], tolerance_ms=tol,
                **{n:scores[n] for n in ('tp','fp','fn','precision','recall','f1')},
                f1_ci95_low=scores['f1_ci95_paired_trajectory_bootstrap'][0],
                f1_ci95_high=scores['f1_ci95_paired_trajectory_bootstrap'][1]))
        stored = model['accuracy']['events']
        assert all(sensitivity['5'][n] == stored[n] for n in ('tp','fp','fn','precision','recall','f1'))
        pooled = math.sqrt(sum(r['voltage']['sse'] for r in records)/sum(r['voltage']['n'] for r in records))
        assert np.isclose(pooled, model['accuracy']['pooled_rmse_mv'], rtol=1e-12)
        count_error = np.array([len(r['predicted_events_ms'])-len(r['reference_events_ms']) for r in records])
        count_bound = sum(min(len(r['reference_events_ms']), len(r['predicted_events_ms'])) for r in records)
        count_summary = dict(mean_signed_error=float(count_error.mean()),
            mean_absolute_error=float(np.abs(count_error).mean()),
            exact_count_trajectories=int((count_error == 0).sum()),
            exact_count_fraction=float((count_error == 0).mean()),
            undercount_trajectories=int((count_error < 0).sum()),
            overcount_trajectories=int((count_error > 0).sum()),
            absolute_count_error_sum=int(np.abs(count_error).sum()),
            within_trajectory_count_only_max_matches=count_bound,
            count_only_f1_upper_bound=2*count_bound/(sum(len(r['reference_events_ms'])+len(r['predicted_events_ms']) for r in records)),
            unmatched_pair_budget_at_5ms=count_bound-sensitivity['5']['tp'])
        assert sensitivity['5']['fp']+sensitivity['5']['fn'] == count_summary['absolute_count_error_sum']+2*count_summary['unmatched_pair_budget_at_5ms']
        zero_stochastic = [r for r in records if not r['reference_events_ms'] and r['family'] != 'zero']
        zero_controls = [r for r in records if r['family'] == 'zero']
        by_family = {}
        for family in ['stationary', 'excitatory_gate', 'rate_steps', 'zero']:
            selected = [i for i, r in enumerate(records) if r['family'] == family]
            if not selected:
                continue
            by_family[family] = dict(cases=len(selected),
                sensitivity={str(t):score_counts(*counts[mi, ti, selected].sum(axis=0)) for ti,t in enumerate(TOLS)})
        # Preserve exact source labels if a future dataset uses another family name.
        for family in sorted({r['family'] for r in records}-set(by_family)):
            selected = [i for i,r in enumerate(records) if r['family'] == family]
            by_family[family] = dict(cases=len(selected),
                sensitivity={str(t):score_counts(*counts[mi, ti, selected].sum(axis=0)) for ti,t in enumerate(TOLS)})
        results[key] = dict(name=NAMES[key], sensitivity=sensitivity,
            reference_events=sum(len(r['reference_events_ms']) for r in records),
            predicted_events=sum(len(r['predicted_events_ms']) for r in records),
            pooled_masked_rmse_mv=pooled,
            per_trajectory_rmse_all379=quantiles([r['voltage']['rmse_mv'] for r in records]),
            per_trajectory_rmse_stochastic374=quantiles([records[i]['voltage']['rmse_mv'] for i in stochastic]),
            no_reference_spikes_stochastic=zero_stats(zero_stochastic),
            no_input_controls=zero_stats(zero_controls),
            count_error=count_summary, by_family=by_family)
        all_errors[key] = errors
    # Collapsing duplicate implementations is justified from complete event lists.
    for duplicate, canonical in [('cpu1','gpu'), ('cpu8','gpu'), ('core1','neuron8')]:
        for a,b in zip(data['models'][duplicate]['records'], data['models'][canonical]['records']):
            assert a['case_id'] == b['case_id']
            assert a['reference_events_ms'] == b['reference_events_ms']
            assert a['predicted_events_ms'] == b['predicted_events_ms']
    paired = {}
    bootstrap_f1 = {}
    for mi,key in enumerate(KEYS):
        bootstrap_f1[key] = {}
        for ti,tol in enumerate(TOLS):
            b = weights @ counts[mi,ti,stochastic] + counts[mi,ti,controls].sum(axis=0)
            bootstrap_f1[key][tol] = 2*b[:,0]/(2*b[:,0]+b[:,1]+b[:,2])
    for key in KEYS[1:]:
        paired[key] = {}
        for tol in TOLS:
            delta = bootstrap_f1['gpu'][tol]-bootstrap_f1[key][tol]
            paired[key][str(tol)] = dict(
                dbnn_minus_baseline_f1=results['gpu']['sensitivity'][str(tol)]['f1']-results[key]['sensitivity'][str(tol)]['f1'],
                percentile_ci95=np.quantile(delta,[.025,.975]).tolist())
    rankings = {str(t):sorted(KEYS, key=lambda k:results[k]['sensitivity'][str(t)]['f1'], reverse=True) for t in TOLS}
    write_csv(OUT/'tolerance_metrics.csv',csv_tolerance)
    write_csv(OUT/'per_trajectory_diagnostics.csv',csv_cases)
    validation = dict(source_sha256=sha(SOURCE), checkpoint_sha256=data['checkpoint_sha256'],
        checkpoint_file_present_and_verified=CHECKPOINT.exists(),
        source_matches_paper=True, cases=379, stochastic_cases=374, zero_input_controls=5,
        reference_events=data['reference_events'], duration_s=2497, per_case_5ms_counts_exactly_reproduced=True,
        published_5ms_f1_and_precision_reproduced=True, pooled_rmse_reproduced=True,
        same_ids_inputs_durations_references_for_all_methods=True,
        independent_dynamic_program_match_count_checks=dp_checks,
        duplicate_implementation_event_lists_identical=True,
        boundary_eps_ms=BOUNDARY_EPS_MS, original_event_times_preserved=True,
        floating_endpoint_audit=endpoint_counts,
        training_runs=0, inference_calls=0, reset_or_threshold_changes=0,
        manuscript_changes_by_analysis_script=0, analysis_execution='local CPU statistics on existing saved predictions',
        bootstrap_replicates=BOOTSTRAPS, bootstrap_seed=SEED,
        bootstrap_unit='paired complete stochastic trajectories; five deterministic controls held fixed',
        uncertainty_scope='input variation for the fixed cell and frozen checkpoints; not training-seed uncertainty')
    after = {str(p):sha(p) for p in protected}
    assert after == before
    validation['protected_inputs_and_manuscript_hashes_unchanged'] = True
    validation['protected_sha256'] = before
    output = dict(protocol=dict(tolerances_ms=TOLS, primary_tolerance_ms=5,
        learned_sampling_ms=1, solver_integration_ms=[.5,.25,.1], reference_integration_ms=data['reference_dt_ms'],
        voltage_exclusion_ms=10, source=str(SOURCE), source_sha256=sha(SOURCE)),
        models=results, rankings=rankings, paired_f1_difference=paired, validation=validation)
    (OUT/'diagnostics.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    # Paper figures are built separately with the manuscript's compact layout.
    print(json.dumps(dict(report=str(OUT/'DBNN_尖峰诊断_20261016.html'),
        five_ms={k:results[k]['sensitivity']['5']['f1'] for k in KEYS},
        paired_dbnn_minus_elm_5ms=paired['elm']['5'],
        dbnn=results['gpu'], validation=validation),ensure_ascii=False,indent=2))



if __name__=='__main__':
    main()
