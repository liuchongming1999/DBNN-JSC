"""Recompute Appendices K/L from saved arrays and original repeated timings."""
from pathlib import Path
import gzip, json, math, subprocess, sys
import numpy as np
from release_paths import REPO, read, digest


def main():
    mapping = read(REPO/'docs/supplement_asset_map_20261027.json')
    for item in mapping['files']:
        assert digest(REPO/item['path']) == item['sha256'], item['path']
    data = json.loads(gzip.decompress((REPO/'paper/evidence/unfiltered_stress_dbnn_records_20261016.json.gz').read_bytes()))
    cases = {r['case_id']:r for r in mapping['cases']}
    manifest = {r['case_id']:r for r in data['input_manifest']}
    original = read(REPO/'workspace/benchmarks/unfiltered_stress_20261016/manifest.json')['cases']
    assert len(original) == 757 and {r['case_id'] for r in original} == set(manifest)
    training = read(REPO/'workspace/benchmarks/accuracy_frontier_20260929/balanced_1ms/training_manifest.json')['cases']
    assert len(training) == 149 and sum(r['split'] == 'fit' for r in training) == 117
    training_hashes = {r['input_sha256'] for r in training}
    assert set(cases) == set(manifest) == {r['case_id'] for r in data['records']}
    assert len(cases) == 757
    maximum_gap = 0.
    gpu_discrepancies = []
    for row in data['records']:
        cid = row['case_id']; paths = cases[cid]; m = manifest[cid]
        assert m['input_sha256'] not in training_hashes, cid
        x = np.load(REPO/paths['input'])
        import hashlib
        assert hashlib.sha256(x['x'].tobytes()).hexdigest() == m['input_sha256'], cid
        assert hashlib.sha256(x['ids'].tobytes()).hexdigest() == m['ids_sha256'], cid
        reference = np.load(REPO/paths['reference'])
        saved = np.load(REPO/paths['accuracy_prediction'])
        gpu = np.load(REPO/paths['gpu_prediction'])
        assert reference['spike_times_ms'].tolist() == row['reference_events_ms'], cid
        assert saved['predicted_events_ms'].tolist() == row['predicted_events_ms'], cid
        rv = reference['voltage'].astype(float); v = saved['voltage'].astype(float)
        assert rv.shape == v.shape == gpu['voltage'].shape, cid
        assert np.isfinite(rv).all() and np.isfinite(v).all() and np.isfinite(gpu['voltage']).all(), cid
        t = np.arange(len(rv), dtype=float)
        assert np.array_equal(reference['time_ms'], t), cid
        assert np.array_equal(saved['time_ms'], t), cid
        keep = np.ones(len(rv), dtype=bool)
        for spike in reference['spike_times_ms']:
            keep &= np.abs(t-spike) > 10.
        error = v-rv
        assert int(keep.sum()) == row['n'] and len(rv) == row['n_all'], cid
        assert math.isclose(float(np.sum(error[keep]**2)), row['sse'], rel_tol=1e-12, abs_tol=1e-8), cid
        assert math.isclose(float(np.sum(error**2)), row['sse_all'], rel_tol=1e-12, abs_tol=1e-8), cid
        gap = float(np.max(np.abs(v-gpu['voltage'])))
        maximum_gap = max(maximum_gap, gap)
        same_events = np.array_equal(saved['predicted_events_ms'], gpu['predicted_events_ms'])
        if row['original_main379']:
            assert gap < .001 and same_events, (cid,gap)
        if gap >= .001 or not same_events:
            gpu_discrepancies.append({'case_id':cid, 'cohort':row['cohort'],
                'maximum_gap_mv':gap, 'events_identical':same_events,
                'accuracy_events':len(saved['predicted_events_ms']),
                'gpu_events':len(gpu['predicted_events_ms'])})
    for name in ['build_unfiltered_stress_appendix.py','build_engineering_ablation_appendix.py']:
        subprocess.run([sys.executable,str(REPO/'paper/scripts'/name),'--check'],check=True)
    engineering = json.loads(gzip.decompress((REPO/'paper/evidence/engineering_ablation_records_20261016.json.gz').read_bytes()))
    for key, folder, source in [('cumulative','engineering','engineering_cumulative.json'),('removal','engineering_rollback','engineering_rollbacks.json')]:
        path = REPO/'workspace/benchmarks/adviser6_mapping_engineering_20261016'/folder/'complete.json'
        assert digest(path) == engineering['source_hashes'][source]
        original = read(path); compact = engineering['campaigns'][key]
        for field in ['protocol','records','summary']:
            assert original[field] == compact[field], (key,field)
    lstm = read(REPO/'paper/evidence/lstm_implementation_clarification_20261027.json')
    for source,path in mapping['lstm_source_map'].items():
        assert digest(REPO/path) == lstm['source_sha256'][source]
    cp = REPO/'workspace/benchmarks/baselines_20260930/aligned/lstm_1ms/best.pt'
    assert digest(cp) == lstm['checkpoint_sha256']
    complete = read(REPO/mapping['lstm_source_map']['source_snapshot/aligned/lstm_1ms/complete.json'])
    assert complete['config'] == lstm['actual_config']
    output = {
        'passed':True, 'supplementary_inputs':757, 'input_and_source_files_hashed':len(mapping['files']),
        'all_supplementary_inputs_disjoint_from_fit_and_validation':True,
        'masked_and_unmasked_sse_recomputed_from_arrays':True,
        'primary379_gpu_parity_passed':True,
        'gpu_accuracy_trace_discrepancies':gpu_discrepancies,
        'gpu_discrepancies_used_to_replace_accuracy_outputs':False,
        'maximum_gpu_accuracy_trace_gap_mv':maximum_gap,
        'engineering_variants':12, 'engineering_inputs':379, 'timed_calls':12*379*30,
        'engineering_original_records_and_source_hashes_match':True,
        'lstm_checkpoint_and_config_unchanged':True,
    }
    destination = REPO/'build/supplement_verification.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(output,indent=2)+'\n')
    brief = {k:v for k,v in output.items() if k != 'gpu_accuracy_trace_discrepancies'}
    brief['gpu_discrepancy_cases'] = len(gpu_discrepancies)
    print(json.dumps(brief),flush=True)
    return output


if __name__ == '__main__':
    main()
