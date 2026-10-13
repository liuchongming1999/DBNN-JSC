"""Replace only DBNN operating points in the fixed paper comparison payload."""
from collections import Counter
import copy
import json
from pathlib import Path
import numpy as np

ROOT = Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
WORK = ROOT / 'benchmarks/random_init_main379_20261012'
PANEL = ROOT / 'benchmarks/spike_panel_20261009'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    d = copy.deepcopy(read(PANEL / 'main379/main379_results.json'))
    evaluation = read(WORK / 'evaluation.json')
    assert evaluation['finished']
    original = d['models']['gpu']['records']
    actual = evaluation['records']
    assert [(r['case_id'], r['input_sha256']) for r in original] == [(r['case_id'], r['input_sha256']) for r in actual]
    for key in ['gpu', 'cpu1', 'cpu8']:
        timing = read(WORK / key / 'complete.json')
        assert timing['finished'] and len(timing['records']) == 379
        assert timing['runtime']['checkpoint_sha256'] == evaluation['checkpoint_sha256']
        assert all(r['validation']['passed'] for r in timing['records'])
        assert [r['input_sha256'] for r in actual] == [r['input_sha256'] for r in timing['records']]
        d['models'][key] = dict(accuracy=evaluation['low_density'], groups=evaluation['groups'],
                              records=actual, timing=timing, interval_ms=1.)
        for duration in [1, 2, 4, 8, 16]:
            values = np.array([r['wall']['median_ms'] for r in timing['records'] if r['duration_s'] == duration])
            d['durations'][str(duration)][key] = dict(cases=len(values), median_ms=float(np.median(values)),
                                                    q1_ms=float(np.quantile(values, .25)),
                                                    q3_ms=float(np.quantile(values, .75)), sum_ms=float(values.sum()))
    d['checkpoint_sha256'] = evaluation['checkpoint_sha256']
    d['dbnn_training_origin'] = 'Random initialization with all quadratic coefficients trainable throughout fitting'
    d['dbnn_evaluation_directory'] = str(WORK)
    selection = []
    neuron = {r['case_id']: r for r in d['models']['neuron8']['records']}
    for n in [0, 3]:
        candidates = [r for r in actual if r['duration_s'] == 8 and r['A'] == 500 and r['reference_spikes'] == n
                      and r['family'] != 'zero' and r['input_events'] > 0
                      and r['events']['tp'] == n and r['events']['fp'] == 0 and r['events']['fn'] == 0
                      and neuron[r['case_id']]['events']['tp'] == n
                      and neuron[r['case_id']]['events']['fp'] == 0
                      and neuron[r['case_id']]['events']['fn'] == 0]
        assert candidates, ('No completely matched 8-s example', n)
        r = min(candidates, key=lambda r: r['voltage']['rmse_mv'])
        selection.append(dict(case_id=r['case_id'], reference_spikes=n, eligible_candidates=len(candidates),
                              dbnn_record=r, neuron_record=neuron[r['case_id']]))
    d['examples'] = selection
    (WORK / 'selected_examples.json').write_text(json.dumps(selection, indent=2) + '\n')
    (WORK / 'main379_results.json').write_text(json.dumps(d, indent=2) + '\n')
    print('BACKENDS', Counter(r['backend'] for r in d['models']['gpu']['timing']['records']))
    print('EXAMPLES', [(r['case_id'], r['eligible_candidates']) for r in selection])
    for key in ['gpu', 'cpu1', 'cpu8']:
        v = d['models'][key]
        print(key, v['accuracy']['pooled_rmse_mv'], v['accuracy']['events']['f1'], v['timing']['wall_ms'])


if __name__ == '__main__':
    main()
