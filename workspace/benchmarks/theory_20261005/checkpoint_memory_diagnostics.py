"""Read-only CPU diagnostics of fixed DBNN filters and history truncation.

Uses the 32 original validation inputs, identical physical event lists for all
three saved short-window checkpoints. It does not train or select a model.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime
import time
import numpy as np
import torch

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n')

def quantiles(x):
    return {str(q): float(np.quantile(x, q)) for q in (0, .25, .5, .75, .9, .95, .99, 1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.root
    checkpoints = {
        1.: root/'benchmarks/window_control_20261004/scratch_matched/w500_burn150/best_checkpoint.pt',
        .5: root/'benchmarks/fair_window_20261005/dbnn/dbnn_w500_dt0.5ms_burn150/best_checkpoint.pt',
        .25: root/'benchmarks/fair_window_20261005/dbnn/dbnn_w500_dt0.25ms_burn150/best_checkpoint.pt',
    }
    manifest = root/'benchmarks/accuracy_frontier_20260929/balanced_1ms/training_manifest.json'
    rows = [r for r in json.loads(manifest.read_text())['cases'] if r['split'] == 'validation']
    starts_ms = np.array([1000., 2000., 3000., 4000., 5000.])
    buffers_ms = np.array([0., 25., 50., 75., 100., 150., 200., 250., 300., 350.])
    result = dict(created=datetime.now().isoformat(), diagnostic='Fixed-model history truncation, drive u only',
        cpu_threads=1, gpu_used=False, training_performed=False,
        cases=len(rows), case_ids=[r['case_id'] for r in rows],
        starts_ms=starts_ms.tolist(), buffers_ms=buffers_ms.tolist(),
        manifest=str(manifest), manifest_sha256=digest(manifest),
        common_input='Same original 1-ms event lists, zero insertion on finer grids; no event jitter', models=[])
    begun = time.perf_counter()
    for dt, checkpoint in checkpoints.items():
        cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
        s = cp['model_state']
        tr = s['tau_rise'].double().numpy().reshape(-1)
        td = s['tau_decay'].double().numpy().reshape(-1)
        omega = s['omega'].double().numpy().reshape(-1)
        w = s['bilinear.weight'].double().numpy()[0].copy()
        np.fill_diagonal(w, 0)
        symmetric = (w+w.T)/2
        wnorm = float(np.abs(np.linalg.eigvalsh(symmetric)).max())
        bias = float(s['bilinear.bias'].reshape(-1)[0])
        reset = {k: float(s[k]) for k in ('v_th', 'reset_amp', 'tau_reset')}
        a = np.exp(-dt/td)
        c = np.exp(-dt*(1/tr+1/td))
        starts = np.rint(starts_ms/dt).astype(int)
        buffer_indices = np.rint(buffers_ms/dt).astype(int)
        selected = (starts[:, None]+buffer_indices[None]).ravel()
        selected_map = {int(n): j for j, n in enumerate(selected)}
        start_map = {int(n)-1: j for j, n in enumerate(starts)}
        fulls, deltas, state_norms = [], [], []
        case_records = []
        for row in rows:
            p = Path(row['input_file'])
            with np.load(p) as inp:
                ids = inp['ids'].astype(int)
                raw = inp['x']
                assert hashlib.sha256(raw.tobytes()).hexdigest() == row['input_sha256']
                x = np.zeros((500, 6000*round(1/dt)+1), np.float64)
                x[ids, ::round(1/dt)] = raw*np.where(ids[:, None] < 250, 1., -1.)
            pa = np.zeros(500)
            pc = np.zeros(500)
            features = np.empty((len(selected), 500))
            histories = np.empty((len(starts), 2, 500))
            for n in range(int(selected.max())+1):
                pa = a*pa+x[:, n]
                pc = c*pc+x[:, n]
                j = selected_map.get(n)
                if j is not None:
                    features[j] = omega*(pa-pc)
                j = start_map.get(n)
                if j is not None:
                    histories[j, 0] = pa
                    histories[j, 1] = pc
            ages = (buffer_indices+1)[:, None]
            delta = np.stack([omega*(a[None]**ages*h[0]-c[None]**ages*h[1])
                              for h in histories]).reshape(-1, 500)
            fulls.append(features)
            deltas.append(delta)
            norms = np.linalg.norm(histories.reshape(len(starts), -1), axis=1)
            state_norms.append(np.repeat(norms, len(buffers_ms)))
            case_records.append(dict(case_id=row['case_id'], input_file=str(p), input_sha256=row['input_sha256']))
        yf = np.concatenate(fulls)
        dy = np.concatenate(deltas)
        yc = yf-dy
        gradient = 1+(yf+yc)@symmetric
        dv = (dy*gradient).sum(1)
        dy_norm = np.linalg.norm(dy, axis=1)
        point_bound = dy_norm*np.linalg.norm(gradient, axis=1)
        coordinate_bound = (np.abs(dy)*np.abs(gradient)).sum(1)
        rmax = float(max(np.linalg.norm(yf, axis=1).max(), np.linalg.norm(yc, axis=1).max()))
        lf = float(np.sqrt(500)+2*rmax*wnorm)
        cw = float(np.sqrt(2)*np.abs(omega).max())
        all_initial = np.concatenate(state_norms)
        bs = np.tile(buffers_ms, len(rows)*len(starts))
        global_bound = lf*cw*all_initial*np.exp(-bs/td.max())
        assert np.max(np.abs(dv)-point_bound) < 1e-8
        assert np.max(np.abs(dv)-coordinate_bound) < 1e-8
        assert np.max(coordinate_bound-point_bound) < 1e-8
        assert np.max(np.abs(dv)-global_bound) < 1e-8
        rec = dict(dt_ms=dt, checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
            m=500, tau_rise_ms=quantiles(tr), tau_decay_ms=quantiles(td), omega=quantiles(omega),
            symmetric_weight_norm=wnorm, bias_mv=bias, reset=reset,
            max_abs_omega=float(np.abs(omega).max()), observed_feature_radius=rmax,
            readout_lipschitz_bound=lf, history_coefficient=cw,
            tail_factor_at_150ms=float(np.exp(-150/td.max())),
            points=len(dv), input_records=case_records, summaries=[])
        for b in buffers_ms:
            keep = bs == b
            e = np.abs(dv[keep])
            rec['summaries'].append(dict(buffer_ms=float(b), points=int(keep.sum()),
                mean_abs_drive_gap_mv=float(e.mean()), median_abs_drive_gap_mv=float(np.median(e)),
                p95_abs_drive_gap_mv=float(np.quantile(e, .95)), max_abs_drive_gap_mv=float(e.max()),
                max_pointwise_upper_bound_mv=float(point_bound[keep].max()),
                max_coordinate_upper_bound_mv=float(coordinate_bound[keep].max()),
                max_exponential_upper_bound_mv=float(global_bound[keep].max()),
                rmse_drive_gap_mv=float(np.sqrt(np.mean(dv[keep]**2)))))
        np.savez_compressed(args.out/f'checkpoint_parameters_dt{dt:g}ms.npz',
            tau_rise=tr, tau_decay=td, omega=omega, weight=w, symmetric_weight=symmetric,
            bias=bias, dt_ms=dt, **reset)
        np.savez_compressed(args.out/f'memory_points_dt{dt:g}ms.npz',
            buffer_ms=bs, drive_gap=dv, feature_difference_norm=dy_norm,
            pointwise_bound=point_bound, exponential_bound=global_bound,
            coordinate_bound=coordinate_bound,
            initial_state_norm=all_initial)
        result['models'].append(rec)
        write(args.out/'checkpoint_memory.json', result)
        print('DONE', dt, 'tau_max', td.max(), 'gap150', rec['summaries'][5]['max_abs_drive_gap_mv'], flush=True)
    result['elapsed_seconds'] = time.perf_counter()-begun
    write(args.out/'checkpoint_memory.json', result)

if __name__ == '__main__':
    main()
