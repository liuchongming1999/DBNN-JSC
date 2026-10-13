"""Export two completely matched 8-s examples for the from-scratch paper model."""
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
WORK = ROOT / 'benchmarks/random_init_main379_20261012'
PANEL = ROOT / 'benchmarks/spike_panel_20261009'


def main():
    selection = json.loads((WORK / 'selected_examples.json').read_text())
    out = WORK / 'examples'
    out.mkdir(exist_ok=True)
    for ex in selection:
        cid = ex['case_id']
        ref = np.load(PANEL / 'confirmatory/references' / f'{cid}_reference.npz')
        pred = np.load(WORK / 'predictions' / f'{cid}.npz')
        nr = np.load(PANEL / 'methods/neuron_dt0.5/raw' / f'neuron_dt0.5_{cid}.npz')
        np.savez_compressed(out / f'{cid}.npz', time_ms=ref['time_ms'],
                            reference_voltage=ref['voltage'], neuron_voltage=nr['voltage'],
                            dbnn_voltage=pred['voltage'], reference_events=ref['spike_times'],
                            neuron_events=nr['spike_times'], dbnn_events=pred['predicted_events_ms'])
        shutil.copy2(ex['dbnn_record']['input_file'], out / f'{cid}_input.npz')
        for dt in [.1, .5]:
            old = PANEL / 'main379/examples' / f'{cid}_native{dt:g}.npz'
            if old.exists():
                shutil.copy2(old, out / old.name)
        print('EXPORTED', cid, flush=True)


if __name__ == '__main__':
    main()
