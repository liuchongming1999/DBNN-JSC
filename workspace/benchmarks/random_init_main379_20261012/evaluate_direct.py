"""Evaluate a validation-locked, from-scratch full-matrix model on the fixed panel."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    args = parser.parse_args()
    root, panel, work = [p.resolve() for p in (args.root, args.panel, args.work)]
    cp = args.checkpoint.resolve()
    assert sha(cp) == args.checkpoint_sha256
    manifest_path = panel / 'main379/manifest.json'
    rows = read(manifest_path)['cases']
    assert len(rows) == 379
    work.mkdir(parents=True, exist_ok=True)
    if (work / 'evaluation.json').exists():
        done = read(work / 'evaluation.json')
        assert done['checkpoint_sha256'] == sha(cp)
        assert done['manifest_sha256'] == sha(manifest_path)
        print('ALREADY_COMPLETE', flush=True)
        return
    sys.path[:0] = [str(panel), str(root / 'benchmarks/diagonal_1ms_20261009'),
                   str(root / 'benchmarks/accuracy_frontier_20260929'),
                   str(root / 'benchmarks/training_refinement_20260929')]
    from train_dbnn_diagonal import VariableDrive, fft_length, load_reset
    from common import metric, aggregate
    from train_refinement import match_events, event_aggregate
    from score_records import grouped
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state = torch.load(cp, map_location='cpu', weights_only=False)['model_state']
    assert torch.diagonal(state['bilinear.weight'][0]).abs().max() > 0
    pp = {key: float(state[key]) for key in ('v_th', 'reset_amp', 'tau_reset')}
    model = VariableDrive(state, torch.device('cuda:0'), 1., True)
    reset = load_reset(root / 'benchmarks/diagonal_1ms_20261009')
    records, folder = [], work / 'predictions'
    folder.mkdir(exist_ok=True)
    write(work / 'protocol.json', dict(
        checkpoint=str(cp), checkpoint_sha256=sha(cp), manifest=str(manifest_path),
        manifest_sha256=sha(manifest_path), source='validation-selected full-matrix run from random initialization',
        checkpoint_and_reset_frozen=True, sample_interval_ms=1., reference_dt_ms=.1,
        voltage_mask_ms=10, event_tolerance_ms=5, repeats_for_timing=3,
        test_inputs_used_for_training_or_selection=False,
        model='one double-exponential kernel per site; quadratic + linear + bias; event reset',
        reset=pp))
    for row in rows:
        cid = row['case_id']
        cached = work / 'records' / f'{cid}.json'
        if cached.exists():
            rec = read(cached)
            assert rec['input_sha256'] == row['input_sha256']
            assert rec['checkpoint_sha256'] == sha(cp)
            records.append(rec)
            continue
        inp = np.load(row['input_file'])
        ids, raw = inp['ids'], inp['x']
        assert hashlib.sha256(raw.tobytes()).hexdigest() == row['input_sha256']
        n = row['duration_ms'] + 1
        dense = np.zeros((500, n), np.float32)
        dense[ids] = raw * np.where(ids[:, None] < 250, 1, -1)
        xf = torch.fft.rfft(torch.from_numpy(dense).cuda()[None], n=fft_length(n))
        drive = model(xf, n)[0].cpu().numpy()
        voltage, events = reset(drive, pp['v_th'], pp['reset_amp'], pp['tau_reset'])
        ref = np.load(panel / 'confirmatory/references' / f'{cid}_reference.npz')
        rr = read(panel / 'confirmatory/references' / f'{cid}_reference.json')
        assert rr['low_density']
        mm = metric(ref['voltage'], voltage, ref['spike_times'], 1.)
        ee = match_events(ref['spike_times'], events)
        rec = dict(**row, reference_spikes=rr['spike_count'], minimum_isi_ms=rr['minimum_isi_ms'],
                   low_density=True, voltage=mm, events=ee,
                   reference_events_ms=ref['spike_times'].tolist(), predicted_events_ms=events.tolist(),
                   checkpoint_sha256=sha(cp))
        np.savez_compressed(folder / f'{cid}.npz', voltage=voltage, drive=drive,
                            time_ms=ref['time_ms'], reference_voltage=ref['voltage'],
                            reference_events_ms=ref['spike_times'], predicted_events_ms=events)
        write(cached, rec)
        records.append(rec)
        write(work / 'evaluation_status.json', dict(completed=len(records), total=len(rows)))
        if len(records) % 20 == 0:
            print('EVALUATE', len(records), len(rows), flush=True)
        del xf
    stats = grouped(records, aggregate, event_aggregate)
    assert stats['low_density']['reference_events'] == 637
    assert stats['low_density']['total_duration_s'] == 2497
    assert sha(cp) == args.checkpoint_sha256
    result = dict(finished=True, created=datetime.now().isoformat(), records=records,
                  checkpoint=str(cp), checkpoint_sha256=sha(cp),
                  manifest_sha256=sha(manifest_path), reset=pp, **stats)
    write(work / 'evaluation.json', result)
    print('COMPLETE', json.dumps(stats['low_density']), flush=True)


if __name__ == '__main__':
    main()
