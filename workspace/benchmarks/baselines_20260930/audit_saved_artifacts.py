"""Inventory reusable main31 trajectories, timings, inputs, and locked weights."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text())


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    work = root/'benchmarks/baselines_20260930'
    manifest_path = root/'benchmarks/dt_resolution_20260929/test/manifest.json'
    manifest = read(manifest_path)
    rows = manifest['cases']
    assert len(rows) == 31 and len({r['case_id'] for r in rows}) == 31
    inventory = dict(created=datetime.now().astimezone().isoformat(), root=str(root),
                     manifest=str(manifest_path), manifest_sha256=digest(manifest_path),
                     cases=[], checkpoints=[], result_sets=[], errors=[])
    for row in rows:
        input_path = Path(row['input_file'])
        with np.load(input_path) as data:
            assert hashlib.sha256(data['x'].tobytes()).hexdigest() == row['input_sha256']
            assert hashlib.sha256(data['ids'].astype(np.int32).tobytes()).hexdigest() == row['ids_sha256']
        reference = manifest_path.parent/'references'/f"{row['case_id']}_reference.npz"
        with np.load(reference) as data:
            assert np.isfinite(data['voltage']).all()
            reference_arrays = {k: dict(shape=list(data[k].shape), dtype=str(data[k].dtype)) for k in data.files}
        inventory['cases'].append(dict(row, reference_path=str(reference),
                                       reference_arrays=reference_arrays))

    def checkpoint(method, dt, path, expected_sha, config):
        actual = digest(path)
        assert actual == expected_sha, (method, dt, 'checkpoint hash mismatch')
        inventory['checkpoints'].append(dict(method=method, dt_ms=dt, path=str(path),
                                             bytes=path.stat().st_size, sha256=actual,
                                             sha256_matches_locked_selection=True, **config))

    def result_set(method, backend, dt, folder, prefix='', native=True,
                   json_folder=None, variant=None):
        saved = []
        for row in rows:
            cid = row['case_id']
            npz = folder/f'{prefix}{cid}.npz'
            rec_path = (json_folder or folder)/f'{prefix}{cid}.json'
            rec = read(rec_path)
            assert rec['input_sha256'] == row['input_sha256']
            if variant:
                timing = rec['variants'][variant]['wall']['runs_ms']
            elif 'backends' in rec:
                timing = rec['backends'][backend]['runs_ms']
            elif 'timing' in rec:
                assert rec['validation']['passed']
                timing = rec['timing']['solver_ms']['runs_ms']
            else:
                timing = rec['wall']['runs_ms']
            assert len(timing) > 0 and all(np.isfinite(timing))
            interval = dt if native else 1.
            with np.load(npz) as data:
                assert len(data['voltage']) == round(row['duration_ms']/interval)+1
                assert all(np.isfinite(data[k]).all() for k in data.files)
                arrays = {k: dict(shape=list(data[k].shape), dtype=str(data[k].dtype)) for k in data.files}
            saved.append(dict(case_id=cid, output_npz=str(npz), record_json=str(rec_path),
                              duration_s=row['duration_s'], A=row['A'], rate_hz=row['rate_hz'],
                              site_mode=row['site_mode'], reference_spikes=row['spike_count'],
                              arrays=arrays, timing_repeats=len(timing)))
        inventory['result_sets'].append(dict(method=method, backend=backend, dt_ms=dt,
                                             saved_voltage_interval_ms=dt if native else 1.,
                                             directory=str(folder), cases=31,
                                             all_arrays_readable_and_finite=True, records=saved))

    variants = {1.: 'B2', .5: 'B6', .25: 'B7'}
    dbnn = root/'benchmarks/resolution_frontier_20260929'
    runtime = read(dbnn/'gpu/runtime.json')
    for dt, variant in variants.items():
        cfg = runtime['variants'][variant]
        checkpoint('DBNN', dt, Path(cfg['checkpoint']), cfg['checkpoint_sha256'],
                   dict(metadata_json=str(dbnn/'gpu/runtime.json')))
        for backend in ['gpu', 'cpu1', 'cpu8']:
            result_set('DBNN', backend, dt, dbnn/backend/variant,
                       json_folder=dbnn/backend, variant=variant)
    original = runtime['variants']['U0']
    checkpoint('DBNN original user checkpoint', 1., Path(original['checkpoint']),
               original['checkpoint_sha256'], dict(metadata_json=str(dbnn/'gpu/runtime.json')))
    for method in ['tcn', 'elm', 'lstm']:
        for dt in variants:
            folder = work/'aligned'/f'{method}_{dt:g}ms'
            lock = read(folder/'complete.json')
            checkpoint(method, dt, folder/'best.pt', lock['checkpoint_sha256'],
                       dict(last_checkpoint=str(folder/'last.pt'),
                            resolved_args=str(folder/'resolved_args.json'),
                            training_history=str(folder/'history.json'),
                            lock_json=str(folder/'complete.json')))
            result_set(method, 'gpu', dt, folder/'publication')
            inventory['result_sets'][-1]['learned_cpu_note'] = (
                'CPU1/CPU8 timing samples, per-case metrics and events are in the same JSON. '
                'Separate CPU voltage arrays are not stored; GPU/CPU numerical checks are recorded.')
    for dt in variants:
        result_set('NEURON multisplit', 'cpu8', dt,
                   root/'benchmarks/neuron_single_expanded_20260929/multisplit_results',
                   prefix=f'nt8_dt{dt:g}_', native=False)
        result_set('CoreNEURON', 'cpu1', dt,
                   root/'benchmarks/core_threading_20260929/main31',
                   prefix=f'nt1_dt{dt:g}_', native=False)
        result_set('Neuron_Reduce', 'cpu1', dt, work/'aligned'/f'reduce_{dt:g}ms', native=False)
    inventory['summary'] = dict(
        main_result_sets=len(inventory['result_sets']),
        main_voltage_trajectories=sum(x['cases'] for x in inventory['result_sets']),
        locked_checkpoints=len(inventory['checkpoints']),
        duration_case_counts=dict(sorted(Counter(r['duration_s'] for r in rows).items())),
        all_inputs_and_checkpoints_match_recorded_sha256=True)
    inventory['reuse'] = dict(
        no_model_run_needed=['Group by duration, sites, frequency, spike count',
                             'RMSE/VE/MAE and alternate masks on saved common grids',
                             'Spike precision/recall and timing tolerance',
                             'Per-case timing median, dispersion and grouped comparisons'],
        inference_or_simulation_needed=['New input sequences, durations or spatial site sets',
                                        'New hardware, thread count, batch size or timing boundary',
                                        'Finer solver recording than saved 1ms observation grid',
                                        'Separate full CPU trajectories for learned baselines'],
        retraining_needed=['New neuron or channel/synapse setting requiring an adapted surrogate',
                            'New trained temporal resolution or changed model/training recipe'])
    output = work/'saved_artifact_inventory.json'
    output.write_text(json.dumps(inventory, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(path=str(output), **inventory['summary']), ensure_ascii=False))


if __name__ == '__main__':
    main()
