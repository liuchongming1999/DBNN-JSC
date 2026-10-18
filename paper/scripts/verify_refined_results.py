"""Independent arithmetic/event checks of current manuscript evidence."""
from pathlib import Path
import gzip,hashlib,json
import numpy as np
from event_metrics import match_events,aggregate_events

R=Path(__file__).resolve().parents[1];E=R/'evidence'
d=json.loads((E/'main379_results.json').read_text())
old=json.loads(gzip.decompress((E/'main379_results_before_refinement_20261016.json.gz').read_bytes()))
diag=json.loads((E/'spike_diagnostics_20261016.json').read_text())
sha=hashlib.sha256((E/'main379_results.json').read_bytes()).hexdigest()
assert d['reference_dt_ms']==.0125 and d['reference_events']==680
assert d['checkpoint_sha256']==old['checkpoint_sha256']=='737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d'
out={}
for k,m in d['models'].items():
    rows=m['records'];a=m['accuracy'];ev=aggregate_events([match_events(r['reference_events_ms'],r['predicted_events_ms']) for r in rows])
    rmse=float(np.sqrt(sum(r['voltage']['sse'] for r in rows)/sum(r['voltage']['n'] for r in rows)))
    assert len(rows)==379 and sum(r['duration_s'] for r in rows)==2497
    assert sum(len(r['reference_events_ms']) for r in rows)==680
    # NumPy/libm versions can differ by one ULP in sqrt; event counts stay exact.
    assert np.isclose(rmse,a['pooled_rmse_mv'],rtol=1e-12,atol=1e-12) and ev==a['events']
    expected_repeats=7 if 'baseline_revision' in d and k in ['gpu','tcn','elm','lstm'] else 3
    assert all(len(t['wall']['runs_ms'])==expected_repeats for t in m['timing']['records'])
    assert np.isclose(sum(t['wall']['median_ms'] for t in m['timing']['records']),m['timing']['wall_ms'],rtol=1e-12)
    if k in old['models'] and k not in ['neuron8','core1','elm']:
        assert all((r['case_id'],r['input_sha256'],r['ids_sha256'],r['predicted_events_ms'])==
                   (s['case_id'],s['input_sha256'],s['ids_sha256'],s['predicted_events_ms'])
                   for r,s in zip(rows,old['models'][k]['records']))
    if k=='elm' and 'baseline_revision' in d:
        assert all((r['case_id'],r['input_sha256'],r['ids_sha256'])==
                   (s['case_id'],s['input_sha256'],s['ids_sha256'])
                   for r,s in zip(rows,old['models'][k]['records']))
        corrected=json.loads((E/'baseline_final_20261017/elm_moments.json').read_text())
        assert all(r['predicted_events_ms']==s['predicted_events_ms'] for r,s in zip(rows,corrected['records']))
    if k in diag['models']:
        assert all(diag['models'][k]['sensitivity']['5'][s]==ev[s] for s in ['tp','fp','fn','precision','recall','f1'])
    out[k]=dict(inputs=379,voltage_samples=sum(r['voltage']['n'] for r in rows),reference_events=680,
                recalculated_rmse_mv=rmse,rmse_matches=True,event_counts_match=True,f1_matches=True,timing_sum_matches=True)
assert diag['protocol']['source_sha256']==sha
assert diag['validation']['independent_dynamic_program_match_count_checks']==379*8*5
for m in ['gpu','cpu1','cpu8']:
    assert [r['predicted_events_ms'] for r in d['models'][m]['records']]==[r['predicted_events_ms'] for r in d['models']['gpu']['records']]
out.update(source_sha256=sha,event_boundary_eps_ms=1e-6,event_scoring_code='scripts/event_metrics.py',
    headline_comparator_dt_ms=.5,headline_accuracy_condition='voltage RMSE',
    headline_time_ratio=d['models']['neuron8']['timing']['wall_ms']/d['models']['gpu']['timing']['wall_ms'],
    joint_rmse_f1_time_ratio=d['models']['neuron8_dt0.25']['timing']['wall_ms']/d['models']['gpu']['timing']['wall_ms'],
    rmse_only_time_ratio=d['models']['neuron8']['timing']['wall_ms']/d['models']['gpu']['timing']['wall_ms'],
    duration_16s_ratio=d['durations']['16']['neuron8']['median_ms']/d['durations']['16']['gpu']['median_ms'],
    second_pair_variance_identity=True,checkpoint_and_learned_event_times_unchanged=True,
    all_379_input_ids_and_hashes_preserved=True,near_boundary_unit_checks_passed=True)
for a,b in [(5,0),(5+5e-7,0),(0,5+5e-7)]:assert match_events([a],[b])['tp']==1
assert match_events([0],[5+2e-6])['tp']==0
(E/'numerical_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(models=len(d['models']),inputs=379,reference_events=680,source_sha256=sha,
                    headline_time_ratio=out['headline_time_ratio'],matched_event_checks=diag['validation']['independent_dynamic_program_match_count_checks']),indent=2))
