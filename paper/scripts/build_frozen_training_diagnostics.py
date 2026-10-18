"""Recheck archived frozen-model diagnostics and generate their three tables.

No training, inference, parameter selection, or primary-result file is modified.
The default mode checks the checked-in tables; --write regenerates only them.
"""
from pathlib import Path
import argparse
import json
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    evidence = root / 'evidence/frozen_training_diagnostics_20261017'
    read = lambda name: json.loads((evidence / name).read_text())
    diagnostic = read('complete.json')
    snapshots = read('snapshot_stability.json')
    parameters = read('validation_grid_parameters.json')
    primary = json.loads((root / 'evidence/main379_results.json').read_text())
    assert diagnostic['finished'] and diagnostic['checkpoint_unchanged']
    assert diagnostic['all_parameters_and_threshold_frozen']
    assert diagnostic['checkpoint_sha256'] == primary['checkpoint_sha256']
    assert snapshots['passed'] and snapshots['final_checkpoint_and_reset_unchanged']
    assert diagnostic['validation_trajectories'] == 32
    assert diagnostic['validation_reference_events'] == 23
    assert diagnostic['bootstrap']['replicates'] == snapshots['replicates'] == 10000

    # Regenerate exactly the original whole-trajectory resamples, including
    # event-free trajectories, then re-evaluate J without any new inference.
    ix = np.random.default_rng(20261017).integers(0, 32, (10000, 32))
    counts = np.array([np.bincount(row, minlength=32) for row in ix], dtype=float)
    grid = np.load(evidence / 'validation_grid_arrays.npz')
    assert grid['sse'].shape == (288, 32) and len(parameters) == 288
    # Explicit reductions avoid platform-specific BLAS floating-status flags.
    aggregate = lambda values: np.einsum('bi,ci->bc', counts, values, optimize=False)
    tp, fp, fn = [aggregate(grid[k]) for k in ['tp', 'fp', 'fn']]
    den = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
    sizes = np.einsum('bi,i->b', counts, grid['size'], optimize=False)
    score = (aggregate(grid['sse']) / sizes[:, None]
             + .02 * aggregate(grid['norm']) / 32 + .05 * (1 - f1))
    assert np.all(np.isfinite(score))
    selected = np.argmin(score, axis=1)
    key = lambda p: tuple(p[k] for k in ['v_th', 'reset_amp', 'tau_reset'])
    assert all(key(parameters[a]) == key(parameters[b]) for a, b in
               zip(selected, grid['bootstrap_selected_indices']))
    reset_frequencies = diagnostic['bootstrap']['reset_frequencies']
    assert sum(row['count'] for row in reset_frequencies) == 10000
    for row in reset_frequencies:
        assert sum(key(parameters[i]) == key(row) for i in selected) == row['count']
        assert row['fraction'] == row['count'] / 10000

    states = np.load(evidence / 'snapshot_validation_arrays.npz')
    assert states['sse'].shape == (6, 32)
    scores = []
    for i, candidate in enumerate(snapshots['candidates']):
        tp, fp, fn = [states[k][i][ix].sum(1) for k in ['tp', 'fp', 'fn']]
        den = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, den, out=np.zeros(10000), where=den > 0)
        scores.append(states['sse'][i][ix].sum(1) / states['n'][ix].sum(1)
                      + .02 * states['normalized'][i][ix].mean(1) + .05 * (1 - f1))
        counts_i = [states[k][i].sum() for k in ['tp', 'fp', 'fn']]
        f1_i = 2 * counts_i[0] / (2 * counts_i[0] + counts_i[1] + counts_i[2])
        rmse_i = np.sqrt(states['sse'][i].sum() / states['n'].sum())
        j_i = rmse_i ** 2 + .02 * states['normalized'][i].mean() + .05 * (1 - f1_i)
        assert abs(j_i - candidate['full_validation_J']) < 1e-12
        assert abs(rmse_i - candidate['pooled_rmse_mv']) < 1e-12
        assert abs(f1_i - candidate['events']['f1']) < 1e-12
    chosen = np.argmin(np.array(scores), axis=0)
    assert np.array_equal(chosen, states['bootstrap_selected_indices'])
    for i, candidate in enumerate(snapshots['candidates']):
        assert np.mean(chosen == i) == candidate['selection_fraction']
    groups = snapshots['equivalent_snapshot_groups']
    assert len(groups) == 4 and abs(sum(g['selection_fraction'] for g in groups) - 1) < 1e-12
    candidates = {c['name']: c for c in snapshots['candidates']}
    for group in groups:
        assert len({candidates[n]['physical_state_digest'] for n in group['names']}) == 1
        assert len({json.dumps(candidates[n]['reset'], sort_keys=True) for n in group['names']}) == 1
        assert abs(sum(candidates[n]['selection_fraction'] for n in group['names'])
                   - group['selection_fraction']) < 1e-12

    for split, expected in [('validation', (32, 23)), ('main379', (379, 680))]:
        result = diagnostic['comparisons'][split]
        assert (result['inputs'], result['reference_spikes']) == expected
        assert len(result['records']) == expected[0]
        assert sum(len(r['reference_events_ms']) for r in result['records']) == expected[1]
        for metric in ['reference_event_reset_voltage', 'autonomous_reset_voltage', 'drive_only_voltage']:
            n = sum(r[metric]['n'] for r in result['records'])
            sse = sum(r[metric]['sse'] for r in result['records'])
            assert n == result[metric]['n']
            assert abs(sse - result[metric]['sse']) < 1e-7
            assert abs(np.sqrt(sse / n) - result[metric]['pooled_rmse_mv']) < 1e-12
        for row in result['records']:
            residual = (row['autonomous_reset_voltage']['sse']
                        - row['reference_event_reset_voltage']['sse']
                        - row['reset_disagreement_sse'] - 2 * row['error_cross_sum'])
            assert abs(residual) < 1e-7
    main_result = diagnostic['comparisons']['main379']
    assert abs(main_result['autonomous_reset_voltage']['pooled_rmse_mv']
               - primary['models']['gpu']['accuracy']['pooled_rmse_mv']) < 1e-12
    assert main_result['events']['tp'] == primary['models']['gpu']['accuracy']['events']['tp']
    assert main_result['events']['fp'] == primary['models']['gpu']['accuracy']['events']['fp']
    assert main_result['events']['fn'] == primary['models']['gpu']['accuracy']['events']['fn']

    tables = {}
    lines = [r'\begin{tabular}{rrrr}', r'\toprule',
             r'$\theta$ (mV) & $a_R$ (mV) & $\tau_R$ (ms) & Selected (\%)\\', r'\midrule']
    for row in reset_frequencies:
        lines.append(f"${row['v_th']:g}$ & ${row['reset_amp']:g}$ & ${row['tau_reset']:g}$ & {100*row['fraction']:.2f}" + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    tables['reset_selection_table.tex'] = '\n'.join(lines) + '\n'
    labels = ['First stage: score/voltage minima', 'First stage: EMA',
              'Second stage: score/voltage minima', 'Second stage: EMA (selected)']
    lines = [r'\begin{tabularx}{\textwidth}{Xrrr}', r'\toprule',
             r'Parameter state & RMSE (mV) & F1 & Selected (\%)\\', r'\midrule']
    for label, group in zip(labels, groups):
        row = candidates[group['names'][0]]
        lines.append(f"{label} & {row['pooled_rmse_mv']:.3f} & {row['events']['f1']:.3f} & {100*group['selection_fraction']:.2f}" + r'\\')
    lines += [r'\bottomrule', r'\end{tabularx}']
    tables['checkpoint_selection_table.tex'] = '\n'.join(lines) + '\n'
    lines = [r'\begin{tabularx}{\textwidth}{Xrrrr}', r'\toprule',
             r'Set & Inputs & Ref. spikes & \multicolumn{2}{c}{RMSE (mV)}\\',
             r' & & & Autonomous & Ref.-event reset\\', r'\midrule']
    for label, split in [('Validation', 'validation'), ('Primary panel', 'main379')]:
        row = diagnostic['comparisons'][split]
        lines.append(f"{label} & {row['inputs']} & {row['reference_spikes']} & {row['autonomous_reset_voltage']['pooled_rmse_mv']:.3f} & {row['reference_event_reset_voltage']['pooled_rmse_mv']:.3f}" + r'\\')
    lines += [r'\bottomrule', r'\end{tabularx}']
    tables['reset_diagnostic_table.tex'] = '\n'.join(lines) + '\n'
    for name, body in tables.items():
        path = root / 'sections' / name
        if args.write:
            path.write_text(body)
        else:
            assert path.read_text() == body, name
    print(json.dumps({'passed': True, 'bootstrap_replicates': 10000,
                      'validation_trajectories': 32, 'validation_reference_spikes': 23,
                      'reset_combinations': 288, 'distinct_checkpoint_states': 4,
                      'tables_checked': list(tables),
                      'checkpoint_sha256': diagnostic['checkpoint_sha256']}, indent=2))


if __name__ == '__main__':
    main()
