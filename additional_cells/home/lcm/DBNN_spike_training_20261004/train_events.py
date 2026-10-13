"""Fixed-architecture event-supervised adaptation; fit/validation only.

The original per-cell checkpoint and all published results remain immutable.
Every trial starts from the same locked per-cell checkpoint and uses identical
data, seed, optimizer and budget. Test traces are read only by a later evaluator.
"""
import argparse
import copy
import ctypes
import fcntl
import hashlib
import json
import math
import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
RESET_LIBRARY = None


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sequential_reset(drive, threshold, amp, tau_samples):
    # Identical recurrence to the event-driven inference reset. FP64 CPU
    # evaluation is outside the inference timer and checked against its oracle.
    global RESET_LIBRARY
    if RESET_LIBRARY is None:
        source = Path(__file__).with_name('calibration_reset.cpp')
        library = source.with_suffix('.so')
        with source.with_suffix('.build.lock').open('w') as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            if not library.exists() or library.stat().st_mtime < source.stat().st_mtime:
                subprocess.run(['g++', '-O3', '-std=c++17', '-fPIC', '-shared', str(source), '-o', str(library)], check=True)
        RESET_LIBRARY = ctypes.CDLL(str(library))
        RESET_LIBRARY.reset_voltage.argtypes = [np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS'),
                                                np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS'),
                                                np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS'),
                                                ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_double]
        RESET_LIBRARY.reset_voltage.restype = ctypes.c_int64
    drive = np.ascontiguousarray(drive, dtype=np.float64)
    voltage = np.empty(len(drive), np.float64)
    events = np.empty(len(drive), np.int64)
    count = RESET_LIBRARY.reset_voltage(drive, voltage, events, len(drive), threshold, amp, tau_samples)
    return voltage, events[:count].astype(float)


def event_loss(pre_reset_voltage, positive_indices, negative_valid, threshold, margin=.15):
    positive = torch.relu(threshold+margin-pre_reset_voltage.flatten()[positive_indices]).square().mean() if positive_indices.numel() else pre_reset_voltage.sum()*0.
    # Four worst non-event points per trace. Exclude reference +/-5 ms and
    # padded tails, so rare false triggers are not diluted by long quiet traces.
    violations = torch.relu(pre_reset_voltage-(threshold-margin)).square()
    violations = torch.where(negative_valid, violations, torch.zeros_like(violations))
    negative = violations.topk(4, dim=1).values.mean()
    return positive, negative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--cell', required=True)
    parser.add_argument('--profile', choices=['ampa', 'ampa_nmda'], required=True)
    parser.add_argument('--dt', type=float, default=1.)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--refinement-epochs', type=int, default=400)
    parser.add_argument('--trials', default='0,0.2,1.0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root, project, work = args.root.resolve(), args.project.resolve(), args.work.resolve()
    dt = args.dt
    assert dt in (1., .5, .25)
    original_folder = project/'results'/args.cell/args.profile/f'dt{dt}ms'
    out = work/('smoke' if args.smoke else 'results')/args.cell/args.profile/f'dt{dt}ms'
    out.mkdir(parents=True, exist_ok=True)
    guard = open(out/'training.lock', 'w')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'final_selection.json').exists():
        print('ALREADY_COMPLETE', str(out), flush=True)
        return
    sys.path[:0] = [str(root/'benchmarks/channel_ablation_20260929'), str(root/'benchmarks/training_refinement_20260929'), str(project)]
    from train_channel_ablation import Drive, reset as event_reset_oracle, teacher_reset
    from train_refinement import match_events, event_aggregate
    from common import metric, aggregate
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device('cuda:0')
    # Numerical control for the CPU calibration implementation.
    rng_check = np.random.default_rng(10404)
    worst = 0.
    for length in (1001, 6001):
        drive = -60.+np.cumsum(rng_check.normal(0., .25, length))
        for threshold, amp, tau in [(-59., 4., 8.), (-58., 12., 40.), (-60., 8., 16.)]:
            expected_v, expected_e = event_reset_oracle(drive, threshold, amp, tau/dt)
            actual_v, actual_e = sequential_reset(drive, threshold, amp, tau/dt)
            assert np.array_equal(expected_e, actual_e)
            worst = max(worst, float(np.max(abs(expected_v-actual_v))))
    assert worst < 1e-8
    initial_path = original_folder/'best_checkpoint.pt'
    archived_selection = read(original_folder/'final_selection.json')
    assert sha(initial_path) == archived_selection['checkpoint_sha256']
    checkpoint = torch.load(initial_path, map_location='cpu', weights_only=False)
    base = checkpoint['model_state']
    assert not any(k.startswith('aux_') for k in base)
    assert checkpoint.get('time_constant_units', 'ms') == 'ms'
    params0 = {k: float(base[k]) for k in ('v_th', 'reset_amp', 'tau_reset')}
    manifest_path = original_folder/'training_manifest.json'
    rows = read(manifest_path)['cases']
    assert all(r['split'] in ('fit', 'validation') for r in rows)
    assert all('/test/' not in r['reference_file'] for r in rows)
    fit = np.asarray([i for i, r in enumerate(rows) if r['split'] == 'fit'])
    val = np.asarray([i for i, r in enumerate(rows) if r['split'] == 'validation'])
    stride = round(1/dt)
    T = round(max(r['duration_ms'] for r in rows)/dt)+1
    nfft = 1 << (2*T-2).bit_length()
    raster = torch.zeros((len(rows), 500, T), dtype=torch.int8, device=device)
    targets, spikes, lengths, negative_masks, event_bins = [], [], [], [], []
    for i, row in enumerate(rows):
        length = round(row['duration_ms']/dt)+1
        with np.load(row['input_file']) as inp:
            step = round(row['input_interval_ms']/dt)
            raw = inp['x'][:, :round(row['duration_ms']/row['input_interval_ms'])+1]
            assert hashlib.sha256(raw.tobytes()).hexdigest() == row['input_sha256']
            dense = np.zeros((500, T), np.int8)
            dense[inp['ids'], :length:step] = raw*(1-2*(inp['ids'][:, None] >= 250))
            raster[i] = torch.from_numpy(dense).to(device)
        with np.load(row['reference_file']) as ref:
            source_dt = float(row['reference_output_interval_ms'])
            sample_step = round(dt/source_dt)
            assert sample_step >= 1 and abs(sample_step*source_dt-dt) < 1e-8
            voltage = ref['voltage'][::sample_step][:length]
            reference_events = ref['spike_times'][ref['spike_times'] <= row['duration_ms']]
        assert len(voltage) == length
        assert len(reference_events) <= 3 and (len(reference_events)<2 or np.diff(reference_events).min() >= 20.)
        padded = np.full(T, voltage[-1], np.float32)
        padded[:length] = voltage
        targets.append(padded)
        spikes.append(reference_events)
        lengths.append(length)
        neg = np.arange(T) < length
        for event in reference_events:
            neg &= abs(np.arange(T)*dt-event) > 5.
        negative_masks.append(neg)
        event_bins.append(np.rint(reference_events/dt).astype(int))
    target = np.asarray(targets)
    voltage_masks = np.asarray([np.arange(T)<length for length in lengths])
    for i, events in enumerate(spikes):
        for event in events:
            voltage_masks[i] &= abs(np.arange(T)*dt-event) > 10.
    variance = np.asarray([target[i, voltage_masks[i]].astype(float).var() for i in range(len(rows))])
    weights = .3+.7/np.maximum(variance, .04)
    weights /= weights[fit].mean()
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    vm = torch.tensor(voltage_masks, device=device)
    nm = torch.tensor(np.asarray(negative_masks), device=device)
    target_tensor = torch.tensor(target, device=device)
    all_candidates = []
    start = time.perf_counter()

    def predict(model):
        with torch.inference_mode():
            return np.concatenate([model(torch.fft.rfft(raster[val[j:j+4]].float(), n=nfft)).cpu().numpy() for j in range(0, len(val), 4)])

    original_rmse = float(archived_selection['selected']['validation']['pooled_rmse_mv'])
    rmse_limit = original_rmse + max(.05, .25*original_rmse)

    def score(drives, params, details=False):
        params = {k: float(np.float32(v)) for k, v in params.items()}
        voltage_metrics, event_metrics, normalized = [], [], []
        for j, i in enumerate(val):
            pred, events = sequential_reset(drives[j, :lengths[i]].astype(float), params['v_th'], params['reset_amp'], params['tau_reset']/dt)
            mm = metric(target[i, :lengths[i]:stride], pred[::stride], spikes[i], 1.)
            voltage_metrics.append(mm)
            event_metrics.append(match_events(spikes[i], events*dt))
            normalized.append(mm['sse']/mm['n']/max(mm['reference_ss']/mm['n'], .04))
        ss = aggregate(voltage_metrics)
        ss['events'] = event_aggregate(event_metrics)
        ss['normalized_mse_floor_0p04'] = float(np.mean(normalized))
        ss['score'] = ss['pooled_rmse_mv']**2+.02*ss['normalized_mse_floor_0p04']+.25*(1-ss['events']['f1'])
        ss['voltage_eligible'] = ss['pooled_rmse_mv'] <= rmse_limit+1e-10
        if details:
            ss['cases'] = [dict(case_id=rows[i]['case_id'], voltage=m, events=e) for i, m, e in zip(val, voltage_metrics, event_metrics)]
        return ss

    def rank(row):
        return (not row['voltage_eligible'], row['score'])

    protocol = dict(created=datetime.now().isoformat(), cell=args.cell, profile=args.profile, dt_ms=dt,
                    fit_cases=len(fit), validation_cases=len(val), fit_reference_spikes=sum(len(spikes[i]) for i in fit),
                    validation_reference_spikes=sum(len(spikes[i]) for i in val), input_manifest=str(manifest_path),
                    input_manifest_sha256=sha(manifest_path), original_checkpoint=str(initial_path), original_checkpoint_sha256=sha(initial_path),
                    architecture='Unchanged: one double-exponential filter/site, zero-diagonal quadratic + linear + bias, original hard event reset',
                    epochs=2 if args.smoke else args.epochs, refinement_epochs=0 if args.smoke else args.refinement_epochs,
                    trials=[float(x) for x in args.trials.split(',')], seed=310049, batch_size=8, lr=.0003,
                    optimizer='AdamW wd1e-6; cosine to 2% LR; clip 1; EMA .995; FP32 IEEE',
                    voltage_loss='Original variance-balanced masked MSE + 0.1 squared physical derivative error; ref AP +/-10ms excluded',
                    event_loss='lambda*(positive threshold hinge at nearest reference event sample + mean per-trace top-4 non-event hinge); margin .15 mV; negatives exclude reference +/-5 ms and padded tails',
                    reset='Teacher reset fixed per stage; validation calibration; rebuilt residual before refinement',
                    calibration='Bias+.025 through +24 mV, empirical validation pre-event levels, old reset; amp 4/8/12 and tau 8/16/40 ms, with local threshold refinement',
                    validation_selection='Min pooled masked MSE + .02 normalized MSE + .25*(1-F1@5ms), among candidates within original validation RMSE + max(.05 mV,25%); old locked candidate retained',
                    voltage_rmse_limit_mv=rmse_limit, test_used=False,
                    cpu_reset_oracle_max_abs_mv=worst, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'), smoke=args.smoke)
    write(out/'protocol.json', protocol)
    write(out/'training_manifest.json', dict(cases=rows))
    original_model = Drive(base, T, device)
    original_model.time = original_model.time.float()*dt
    original_drives = predict(original_model)
    original_score = score(original_drives, params0, True)
    assert abs(original_score['pooled_rmse_mv']-original_rmse) < 2e-4
    assert all(original_score['events'][k] == archived_selection['selected']['validation']['events'][k] for k in ('tp', 'fp', 'fn'))
    print('BASELINE', args.cell, args.profile, original_score['pooled_rmse_mv'], original_score['events'], flush=True)

    def save_candidate(model, name, params, validation, source):
        params = {k: float(np.float32(v)) for k, v in params.items()}
        folder = out/'candidates'/name
        folder.mkdir(parents=True, exist_ok=True)
        path = folder/'checkpoint.pt'
        state = model.original_state(base, params)
        torch.save(dict(model_state=state, sample_interval_ms=dt, time_constant_units='ms', protocol=protocol, candidate=name), path)
        candidate = dict(name=name, checkpoint=str(path), checkpoint_sha256=sha(path), reset=params.copy(),
                         validation=validation, source=source)
        write(folder/'validation.json', candidate)
        all_candidates.append(candidate)
        print('CANDIDATE', name, 'RMSE', round(validation['pooled_rmse_mv'], 5), 'F1', round(validation['events']['f1'], 4), 'eligible', validation['voltage_eligible'], flush=True)
        return candidate

    save_candidate(original_model, 'old_locked', params0, original_score, 'Unchanged original weights and reset')

    def calibrate(model, name, current_params, source):
        drives = predict(model)
        bias = float(model.bias.detach())
        thresholds = [bias+x for x in [.025, .05, .1, .2, .4, .7, 1., 1.5, 2., 3., 4., 6., 8., 12., 16., 20., 24.]]
        thresholds.append(current_params['v_th'])
        pre_event_levels = []
        for j, i in enumerate(val):
            recovery = teacher_reset(T, spikes[i]/dt, current_params['reset_amp'], current_params['tau_reset']/dt)
            for k in event_bins[i]:
                pre_event_levels.append(float(drives[j, k]+recovery[k]))
        if pre_event_levels:
            for level in np.quantile(pre_event_levels, [.1, .25, .5, .75, .9]):
                thresholds += [float(level+x) for x in [-.15, 0., .15] if bias+.005 <= level+x <= bias+30.]
        thresholds = sorted(set(round(t, 6) for t in thresholds))
        grid = []
        for th in thresholds:
            for amp in (4., 8., 12.):
                for tau in (8., 16., 40.):
                    params = dict(v_th=th, reset_amp=amp, tau_reset=tau)
                    grid.append(dict(params=params, **score(drives, params)))
        grid.append(dict(params=current_params.copy(), **score(drives, current_params)))
        winner = min(grid, key=rank)
        for delta in [-.25, -.125, -.0625, -.025, 0., .025, .0625, .125, .25]:
            params = {**winner['params'], 'v_th': winner['params']['v_th']+delta}
            if params['v_th'] >= bias+.005:
                grid.append(dict(params=params, **score(drives, params)))
        winner = min(grid, key=rank)
        candidate = save_candidate(model, name, winner['params'], score(drives, winner['params'], True), source)
        write(Path(candidate['checkpoint']).with_name('reset_grid.json'), grid)
        return candidate

    calibrated = calibrate(original_model, 'expanded_calibration_only', params0, 'Old locked weights; new validation-only threshold search')
    # All three loss strengths start from the same weights and calibrated reset.
    params_start = calibrated['reset'].copy()
    initial_state = original_model.original_state(base, params_start)
    del original_model
    smoke_epochs = 2 if args.smoke else args.epochs
    smoke_refine = 0 if args.smoke else args.refinement_epochs
    for strength in protocol['trials']:
        trial_name = f'event_lambda_{strength:g}'
        trial_folder = out/'training'/trial_name
        trial_folder.mkdir(parents=True, exist_ok=True)
        if (trial_folder/'complete.json').exists():
            for c in read(trial_folder/'complete.json')['candidates']:
                all_candidates.append(c)
            continue
        torch.manual_seed(protocol['seed'])
        rng = np.random.default_rng(protocol['seed'])
        model = Drive(initial_state, T, device)
        model.time = model.time.float()*dt
        params = params_start.copy()
        trial_candidates = []
        for stage, epochs, lr in [('primary', smoke_epochs, .0003), ('refine', smoke_refine, .00006)]:
            if epochs == 0:
                continue
            if stage == 'refine':
                best = min(trial_candidates, key=lambda r: rank(r['validation']))
                params = best['reset'].copy()
                state = torch.load(best['checkpoint'], map_location='cpu', weights_only=False)['model_state']
                model = Drive(state, T, device)
                model.time = model.time.float()*dt
            teacher = np.asarray([teacher_reset(T, s/dt, params['reset_amp'], params['tau_reset']/dt) for s in spikes], np.float32)
            teacher_tensor = torch.tensor(teacher, device=device)
            drive_target = target_tensor-teacher_tensor
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=lr*.02)
            ema = copy.deepcopy(model.state_dict())
            history = []
            best_joint, best_voltage = (True, float('inf')), float('inf')
            saved = {}
            for epoch in range(epochs+1):
                losses, event_positive, event_negative = [], [], []
                if epoch:
                    for ix in np.array_split(rng.permutation(fit), math.ceil(len(fit)/8)):
                        optimizer.zero_grad(set_to_none=True)
                        drive = model(torch.fft.rfft(raster[ix].float(), n=nfft))
                        error = drive-drive_target[ix]
                        w = vm[ix]
                        dw = w[:, 1:] & w[:, :-1]
                        per = ((error.square()*w).sum(1)/w.sum(1))+.1*((((error[:, 1:]-error[:, :-1])/dt).square()*dw).sum(1)/dw.sum(1))
                        voltage_loss = (per*weight_tensor[ix]).mean()
                        positive_bins = [b*T+int(k) for b, i in enumerate(ix) for k in event_bins[i]]
                        pos_index = torch.tensor(positive_bins, dtype=torch.long, device=device)
                        pos, neg = event_loss(drive+teacher_tensor[ix], pos_index, nm[ix], params['v_th'])
                        loss = voltage_loss+strength*(pos+neg)
                        if not torch.isfinite(loss):
                            raise RuntimeError(('nonfinite loss', trial_name, stage, epoch))
                        loss.backward()
                        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                        if not torch.isfinite(grad):
                            raise RuntimeError(('nonfinite gradient', trial_name, stage, epoch))
                        optimizer.step()
                        with torch.no_grad():
                            model.log_tau_rise.clamp_(math.log(.25), math.log(50.))
                            model.log_tau_decay.clamp_(math.log(.5), math.log(100.))
                            for key, value in model.state_dict().items():
                                if key not in ('time', 'offdiag'):
                                    ema[key].mul_(.995).add_(value, alpha=.005)
                        losses.append(float(loss.detach()))
                        event_positive.append(float(pos.detach()))
                        event_negative.append(float(neg.detach()))
                    scheduler.step()
                if epoch%20 == 0 or epoch == epochs:
                    validation = score(predict(model), params)
                    record = dict(trial=trial_name, stage=stage, epoch=epoch, fit_loss=float(np.mean(losses)) if losses else None,
                                  event_positive_loss=float(np.mean(event_positive)) if event_positive else None,
                                  event_negative_loss=float(np.mean(event_negative)) if event_negative else None,
                                  validation=validation, elapsed_s=time.perf_counter()-start)
                    history.append(record)
                    if rank(validation) < best_joint:
                        best_joint = rank(validation)
                        saved['joint'] = epoch
                        torch.save(model.state_dict(), trial_folder/(stage+'_joint.pt'))
                    if validation['pooled_rmse_mv'] < best_voltage:
                        best_voltage = validation['pooled_rmse_mv']
                        saved['voltage'] = epoch
                        torch.save(model.state_dict(), trial_folder/(stage+'_voltage.pt'))
                    write(trial_folder/(stage+'_history.json'), history)
                    write(out/'progress.json', record)
                    print('EPOCH', args.cell, args.profile, trial_name, stage, epoch, 'RMSE', round(validation['pooled_rmse_mv'], 5), 'F1', round(validation['events']['f1'], 4), 's', round(record['elapsed_s'], 1), flush=True)
            torch.save(ema, trial_folder/(stage+'_ema.pt'))
            saved['ema'] = epochs
            for tag in ('joint', 'voltage', 'ema'):
                model.load_state_dict(torch.load(trial_folder/(stage+'_'+tag+'.pt'), map_location=device, weights_only=False))
                trial_candidates.append(calibrate(model, trial_name+'_'+stage+'_'+tag, params,
                                                  dict(event_strength=strength, stage=stage, epoch=saved[tag], tag=tag)))
        write(trial_folder/'complete.json', dict(finished=True, event_strength=strength, candidates=trial_candidates))
        del model, optimizer
        torch.cuda.empty_cache()
    winner = min(all_candidates, key=lambda c: rank(c['validation']))
    chosen = torch.load(winner['checkpoint'], map_location='cpu', weights_only=False)
    # Re-load exactly the exported FP32 state before recording locked metrics.
    exported = Drive(chosen['model_state'], T, device)
    exported.time = exported.time.float()*dt
    exported_reset = {k: float(chosen['model_state'][k]) for k in ('v_th', 'reset_amp', 'tau_reset')}
    locked_validation = score(predict(exported), exported_reset, True)
    export_comparison = dict(rmse_difference_mv=locked_validation['pooled_rmse_mv']-winner['validation']['pooled_rmse_mv'],
                             event_counts_equal=all(locked_validation['events'][k] == winner['validation']['events'][k] for k in ('tp', 'fp', 'fn')))
    assert abs(export_comparison['rmse_difference_mv']) < .001
    winner = {**winner, 'validation_before_export_reload': winner['validation'], 'validation': locked_validation}
    chosen['locked_before_test_at'] = datetime.now().isoformat()
    torch.save(chosen, out/'best_checkpoint.pt')
    final = dict(finished=True, selected=winner, candidates=all_candidates, protocol=protocol,
                 checkpoint_sha256=sha(out/'best_checkpoint.pt'), locked_at=chosen['locked_before_test_at'],
                 elapsed_s=time.perf_counter()-start, original_validation=original_score, exported_state_check=export_comparison)
    write(out/'final_selection.json', final)
    print('LOCKED', args.cell, args.profile, winner['name'], winner['validation']['pooled_rmse_mv'], winner['validation']['events'], flush=True)


if __name__ == '__main__':
    main()
