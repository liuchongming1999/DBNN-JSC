"""Paired random-init DBNN controls on the corresponding shared 6-s panels.

The only factorial choices are training window length and loss burn-in.
There is no trained checkpoint initialization and no held-out selection.
"""
import argparse
import copy
import ctypes as C
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime

import numpy as np
import torch
from torch import nn


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().numpy().tobytes())
    return h.hexdigest()


def fft_length(n):
    return 1 << (2 * n - 2).bit_length()


class VariableDrive(nn.Module):
    """Original one-kernel DBNN; T-dependent FFT buffers do not train."""
    def __init__(self, state, device, dt=1.):
        super().__init__()
        self.dt = dt
        for k in ('tau_rise', 'tau_decay'):
            setattr(self, 'log_' + k, nn.Parameter(state[k].to(device).log().clone()))
        self.omega = nn.Parameter(state['omega'].to(device).clone())
        self.weight = nn.Parameter(state['bilinear.weight'][0].to(device).clone())
        self.bias = nn.Parameter(state['bilinear.bias'].to(device).clone())
        self.register_buffer('offdiag', 1 - torch.eye(500, device=device))
        self.times = {}

    def forward(self, xf, n):
        if n not in self.times:
            self.times[n] = torch.arange(n, dtype=torch.float32, device=xf.device) * self.dt
        t = self.times[n]
        tr, td = self.log_tau_rise.exp()[:, None], self.log_tau_decay.exp()[:, None]
        k = self.omega[:, None] * (1 - torch.exp(-t / tr)) * torch.exp(-t / td)
        y = torch.fft.irfft(xf * torch.fft.rfft(k, n=fft_length(n))[None],
                            n=fft_length(n))[..., :n].transpose(1, 2).contiguous()
        return ((y @ (self.weight * self.offdiag)) * y).sum(-1) + y.sum(-1) + self.bias

    def physical_state(self, reset_params):
        s = {k: getattr(self, 'log_' + k).detach().exp().cpu().clone()
             for k in ('tau_rise', 'tau_decay')}
        s.update(omega=self.omega.detach().cpu().clone())
        s['bilinear.weight'] = (self.weight * self.offdiag).detach().cpu()[None].clone()
        s['bilinear.bias'] = self.bias.detach().cpu().clone()
        s.update({k: torch.tensor(v, dtype=torch.float32) for k, v in reset_params.items()})
        return s


def random_state(seed):
    torch.manual_seed(seed)
    w = torch.empty(1, 500, 500).uniform_(-1 / math.sqrt(500), 1 / math.sqrt(500))
    w[0].fill_diagonal_(0)
    return {'tau_rise': torch.full((500,), 5.), 'tau_decay': torch.full((500,), 20.),
            'omega': torch.full((500,), .3), 'bilinear.weight': w,
            'bilinear.bias': torch.tensor([-70.]), 'v_th': torch.tensor(-55.),
            'reset_amp': torch.tensor(15.), 'tau_reset': torch.tensor(10.)}


def load_reset(folder):
    lib = C.CDLL(str(folder / 'seq_reset.so'))
    fn = lib.dbnn_reset
    fn.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t,
                   C.c_double, C.c_double, C.c_double]
    fn.restype = C.c_size_t

    def run(drive, theta, amp, tau):
        d = np.ascontiguousarray(drive, dtype=np.float64)
        v, ev = np.empty_like(d), np.empty_like(d)
        n = fn(d.ctypes.data, v.ctypes.data, ev.ctypes.data, len(d), theta, amp, tau)
        return v, ev[:n]
    return run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--window', required=True, type=int, choices=[400, 500, 6001])
    p.add_argument('--burn', required=True, type=int, choices=[0, 150])
    p.add_argument('--dt', type=float, choices=[1., .5, .25], default=1.)
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--refine-epochs', type=int, default=400)
    p.add_argument('--lr', type=float, default=.003)
    p.add_argument('--seed', type=int, default=291049)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    root, work = a.root.resolve(), a.work.resolve()
    name = f'dbnn_w{a.window}_dt{a.dt:g}ms_burn{a.burn}' + ('_smoke' if a.smoke else '')
    out = work / name
    out.mkdir(parents=True, exist_ok=True)
    guard = open(out / 'training.lock', 'w')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out / 'complete.json').exists():
        print('ALREADY_COMPLETE', name, flush=True)
        return
    if (out / 'protocol.json').exists():
        raise RuntimeError('Unfinished run exists; inspect before overwriting')
    sys.path[:0] = [str(root / 'benchmarks/accuracy_frontier_20260929'),
                   str(root / 'benchmarks/channel_ablation_20260929'),
                   str(root / 'benchmarks/training_refinement_20260929')]
    from common import metric, aggregate, mask
    from train_channel_ablation import teacher_reset, reset as old_reset, Drive
    from train_refinement import match_events, event_aggregate
    reset = load_reset(work)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    initial = random_state(a.seed)
    rng = np.random.default_rng(a.seed)
    params = {k: float(initial[k]) for k in ('v_th', 'reset_amp', 'tau_reset')}
    dt = a.dt
    stride = round(1 / dt)
    burn = round(a.burn / dt)
    torch.save({'model_state': initial, 'sample_interval_ms': dt,
                'time_constant_units': 'ms'}, out / 'random_initial_checkpoint.pt')
    sys.path.insert(0,str(root/'benchmarks/baselines_20260930'))
    from panel import rows_for
    _, manifest = rows_for(root, dt)
    rows = read(manifest)['cases']
    if a.smoke:
        rows = [r for r in rows if r['split'] == 'fit'][:3] + [r for r in rows if r['split'] == 'validation'][:2]
    fit = np.array([i for i, r in enumerate(rows) if r['split'] == 'fit'])
    val = np.array([i for i, r in enumerate(rows) if r['split'] == 'validation'])
    T = 6000 * stride + 1
    W = T if a.window == 6001 else round(a.window / dt)
    xc = torch.empty((len(rows), 500, T), dtype=torch.int8, device=device)
    targets, spikes = [], []
    input_digest = hashlib.sha256()
    for i, r in enumerate(rows):
        assert r['duration_ms'] == 6000
        inp = np.load(r['input_file'])
        ids, raw = inp['ids'], inp['x']
        assert hashlib.sha256(raw.tobytes()).hexdigest() == r['input_sha256']
        input_digest.update(r['case_id'].encode())
        input_digest.update(raw.tobytes())
        dense = np.zeros((500, T), np.int8)
        signed = raw * np.where(ids[:, None] < 250, 1, -1)
        assert np.array_equal(signed, signed.astype(np.int8))
        step = round(r['input_interval_ms'] / dt)
        assert step >= 1 and abs(step * dt - r['input_interval_ms']) < 1e-9
        dense[ids, ::step] = signed
        xc[i] = torch.from_numpy(dense).to(device)
        ref = np.load(r['reference_file'])
        vv = np.interp(np.arange(T)*dt, ref['time_ms'], ref['voltage']).astype(np.float32)
        targets.append(vv)
        spikes.append(ref['spike_times'])
    target = np.stack(targets)
    masks = np.stack([mask(T, s, dt) for s in spikes])
    wm = torch.from_numpy(masks.astype(np.float32)).to(device)
    variances = np.array([np.var(target[i, masks[i]].astype(float)) for i in range(len(rows))])
    weights = .3 + .7 / np.maximum(variances, .04)
    weights /= np.mean(weights[fit])
    cw = torch.tensor(weights, dtype=torch.float32, device=device)
    # Full FFT cache is only needed for validation and the 6-s training control.
    cache_ids = np.arange(len(rows)) if W == T else val
    full_xf = torch.empty((len(cache_ids), 500, fft_length(T) // 2 + 1),
                          dtype=torch.complex64, device=device)
    cache_lookup = {int(k): j for j, k in enumerate(cache_ids)}
    for j, k in enumerate(cache_ids):
        full_xf[j] = torch.fft.rfft(xc[int(k)].float(), n=fft_length(T))
    model = VariableDrive(initial, device, dt)
    # Check the variable-window implementation against the existing DBNN.
    old = Drive(initial, T, device)
    old.time = old.time.float() * dt
    with torch.inference_mode():
        x = full_xf[:1]
        forward_gap = float((old(x) - model(x, T)).abs().max())
    assert forward_gap == 0., forward_gap
    del old
    prng = np.random.default_rng(103)
    reset_gap = 0.
    for th in [-75., -60., -55., 20.]:
        d = prng.normal(-62, 8, min(T,6001))
        vo, eo = old_reset(d, th, 4., 40./dt)
        vn, en = reset(d, th, 4., 40./dt)
        assert np.array_equal(eo, en)
        reset_gap = max(reset_gap, float(np.max(np.abs(vo - vn))))
    assert reset_gap < 1e-10
    protocol = dict(created=datetime.now().isoformat(), name=name, smoke=a.smoke,
        dt_ms=dt, source_trajectory_ms=6000, training_window_samples=W,
        training_window_ms=6000 if W == T else a.window, loss_burn_in_ms=a.burn,
        fit_cases=len(fit), validation_cases=len(val), batch_size=8,
        primary_epochs=a.epochs, refinement_epochs=a.refine_epochs, learning_rate=a.lr, seed=a.seed,
        kernels_per_site=1,
        architecture='Unchanged one double-exponential filter per site, zero-diagonal dense quadratic, linear sum, bias, event reset',
        initialization='Independent random weights; tau_rise=5ms, tau_decay=20ms, omega=.3, bias=-70mV, reset=(-55mV,15mV,10ms). No learned checkpoint.',
        initial_state_digest=state_digest(initial), manifest=str(manifest),
        manifest_sha256=sha(manifest), ordered_input_digest=input_digest.hexdigest(),
        fit_case_ids=[rows[i]['case_id'] for i in fit], validation_case_ids=[rows[i]['case_id'] for i in val],
        window_sampling='Same seeded fit-case permutation, one crop per case/epoch; crop end uniformly sampled after500ms on native grid. 400/500ms arms share ends within each resolution. Each resolution uses exactly its external-baseline source panel.',
        initial_window_state='Zero local filter and reset states; no unobserved input or inherited teacher-reset state before the crop.',
        loss='Same balanced B2 loss: mean case-weighted masked voltage MSE + .1 first-difference MSE. Weights .3+.7/max(full-reference variance,.04), normalized over fit. Reference AP +/-10ms excluded. Burn applied equally to voltage and derivative terms.',
        optimizer=f'AdamW lr {a.lr} / {a.lr*.2} refinement, wd 1e-6; cosine to .02 initial lr; clip1; EMA .995; FP32 IEEE',
        budget='Same case presentations / optimizer updates for the400/500ms arms within each resolution; corresponding resolution-specific fit panel sizes differ. Supervised physical time per update differs with window length.',
        selection='Full continuous 6s validation: pooled masked MSE + .02 mean normalized MSE with .04 variance floor + .05*(1-event F1 at +/-5ms). Best joint, best voltage and EMA candidates, reset grid calibrated on validation only.',
        reset_training='Fixed teacher reset during primary; validation-only reset calibration then refine using rebuilt teachers.',
        tests_used_for_training_or_selection=False, physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        validation_forward_max_gap_mv=forward_gap, sequential_reset_max_gap_mv=reset_gap,
        script_sha256=sha(Path(__file__)), supervised_grid=f'Native{dt}ms; held-out metrics on common1ms with reference AP +/-10ms excluded')
    write(out / 'protocol.json', protocol)
    write(out / 'training_manifest.json', dict(cases=rows))
    start_time = time.perf_counter()
    candidates, updates = [], 0

    def predict(ids):
        with torch.inference_mode():
            return np.concatenate([model(full_xf[[cache_lookup[int(i)] for i in ids[j:j+4]]], T).cpu().numpy()
                                   for j in range(0, len(ids), 4)])

    def score(drives, pp, detail=False):
        ms, es, norms = [], [], []
        for j, i in enumerate(val):
            vv, ev = reset(drives[j], pp['v_th'], pp['reset_amp'], pp['tau_reset']/dt)
            mm = metric(target[i,::stride], vv[::stride], spikes[i], 1.)
            ms.append(mm)
            es.append(match_events(spikes[i], ev*dt))
            norms.append((mm['sse'] / mm['n']) / max(mm['reference_ss'] / mm['n'], .04))
        result = aggregate(ms)
        result.update(events=event_aggregate(es), normalized_mse_floor_0p04=float(np.mean(norms)))
        result['score'] = result['pooled_rmse_mv']**2 + .02 * result['normalized_mse_floor_0p04'] + .05 * (1 - result['events']['f1'])
        if detail:
            result['cases'] = [dict(case_id=rows[i]['case_id'], voltage=m, events=e) for i, m, e in zip(val, ms, es)]
        return result

    def calibrate(tag, epoch):
        drives = predict(val)
        grid = []
        for theta in list(np.arange(-60., -53.9, .5)) + [params['v_th']]:
            for amp in [4., 6., 8., 10., 12.]:
                for tau in [8., 16., 24., 40.]:
                    pp = dict(v_th=float(theta), reset_amp=amp, tau_reset=tau)
                    grid.append(dict(params=pp, **score(drives, pp)))
        grid.append(dict(params=params.copy(), **score(drives, params)))
        best = min(grid, key=lambda r: r['score'])
        for delta in [-.375, -.25, -.125, 0., .125, .25, .375]:
            pp = {**best['params'], 'v_th': best['params']['v_th'] + delta}
            grid.append(dict(params=pp, **score(drives, pp)))
        winner = min(grid, key=lambda r: r['score'])
        folder = out / 'candidates' / tag
        folder.mkdir(parents=True, exist_ok=True)
        cp = folder / 'checkpoint.pt'
        torch.save(dict(model_state=model.physical_state(winner['params']),
                        sample_interval_ms=dt, time_constant_units='ms', protocol=protocol), cp)
        rec = dict(name=tag, epoch=epoch, checkpoint=str(cp), checkpoint_sha256=sha(cp),
                   reset=winner['params'], validation=score(drives, winner['params'], True))
        write(folder / 'validation.json', rec)
        write(folder / 'reset_grid.json', grid)
        candidates.append(rec)
        print('CANDIDATE', tag, rec['validation']['pooled_rmse_mv'], rec['validation']['events']['f1'], flush=True)

    channels = torch.arange(500, device=device)
    offset = torch.arange(W, device=device)
    for stage, epochs, lr in [('primary', a.epochs, a.lr), ('refine', a.refine_epochs, a.lr*.2)]:
        if stage == 'refine':
            best = min(candidates, key=lambda c: c['validation']['score'])
            params = best['reset'].copy()
            model = VariableDrive(torch.load(best['checkpoint'], map_location='cpu', weights_only=False)['model_state'], device, dt)
        teacher = np.stack([teacher_reset(T, s/dt, params['reset_amp'], params['tau_reset']/dt) for s in spikes]).astype(np.float32)
        rt = torch.from_numpy(teacher).to(device)
        y = torch.from_numpy(target - teacher).to(device)
        powers = torch.exp(-offset.float()*dt / params['tau_reset'])
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr*.02)
        ema = copy.deepcopy(model.state_dict())
        history, bestjoint, bestvoltage, saved = [], float('inf'), float('inf'), {}
        for epoch in range(epochs + 1):
            epoch_losses = []
            if epoch:
                for ii in np.array_split(rng.permutation(fit), math.ceil(len(fit) / 8)):
                    # Keep random stream identical for all six factorial arms.
                    ends = rng.integers(round(500/dt), T, len(ii))
                    starts = np.zeros(len(ii), np.int64) if W == T else ends - W
                    ix = torch.tensor(ii, device=device)
                    starts = torch.tensor(starts, device=device)
                    tt = starts[:, None] + offset[None]
                    if W == T:
                        inp = full_xf[[cache_lookup[int(i)] for i in ii]]
                    else:
                        chunk = xc[ix[:, None, None], channels[None, :, None], tt[:, None, :]].float()
                        inp = torch.fft.rfft(chunk, n=fft_length(W))
                    desired = y[ix[:, None], tt] + rt[ix, starts][:, None] * powers[None]
                    weight = wm[ix[:, None], tt]
                    weight[:, :burn] = 0
                    dw = torch.minimum(weight[:, 1:], weight[:, :-1])
                    opt.zero_grad(set_to_none=True)
                    error = model(inp, W) - desired
                    per = (error.square()*weight).sum(1)/weight.sum(1).clamp_min(1)
                    per += .1*(((error[:, 1:]-error[:, :-1])/dt).square()*dw).sum(1)/dw.sum(1).clamp_min(1)
                    loss = (per*cw[ix]).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError(('nonfinite_loss', stage, epoch))
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    assert torch.isfinite(norm)
                    opt.step()
                    with torch.no_grad():
                        model.log_tau_rise.clamp_(math.log(.25), math.log(50.))
                        model.log_tau_decay.clamp_(math.log(.5), math.log(100.))
                        for k, v in model.state_dict().items():
                            if k != 'offdiag':
                                ema[k].mul_(.995).add_(v, alpha=.005)
                    epoch_losses.append(float(loss.detach()))
                    updates += 1
                schedule.step()
            if epoch % 20 == 0 or epoch == epochs:
                ss = score(predict(val), params)
                rec = dict(stage=stage, epoch=epoch, loss=float(np.mean(epoch_losses)) if epoch_losses else None,
                           validation=ss, updates=updates, elapsed_s=time.perf_counter()-start_time)
                history.append(rec)
                if ss['score'] < bestjoint:
                    bestjoint = ss['score']
                    saved['joint'] = epoch
                    torch.save(model.state_dict(), out / f'{stage}_joint.pt')
                if ss['pooled_rmse_mv'] < bestvoltage:
                    bestvoltage = ss['pooled_rmse_mv']
                    saved['voltage'] = epoch
                    torch.save(model.state_dict(), out / f'{stage}_voltage.pt')
                write(out / f'{stage}_history.json', history)
                write(out / 'progress.json', rec)
                print('EPOCH', stage, epoch, 'RMSE', round(ss['pooled_rmse_mv'], 5), 'F1', round(ss['events']['f1'], 4), 's', round(rec['elapsed_s'], 1), flush=True)
        torch.save(ema, out / f'{stage}_ema.pt')
        saved['ema'] = epochs
        for tag in ['joint', 'voltage', 'ema']:
            model.load_state_dict(torch.load(out / f'{stage}_{tag}.pt', map_location=device, weights_only=False))
            calibrate(stage + '_' + tag, saved[tag])
    winner = min(candidates, key=lambda c: c['validation']['score'])
    locked = datetime.now().isoformat()
    chosen = torch.load(winner['checkpoint'], map_location='cpu', weights_only=False)
    chosen['locked_before_test_at'] = locked
    cp = out / 'best_checkpoint.pt'
    torch.save(chosen, cp)
    selection = dict(finished=True, selected=winner, candidates=candidates, protocol=protocol,
                     locked_at=locked, checkpoint_sha256=sha(cp), updates=updates,
                     training_elapsed_s=time.perf_counter()-start_time)
    write(out / 'final_selection.json', selection)
    # Only now open the unchanged held-out panel.
    records = []
    if not a.smoke:
        test_manifest = root / 'benchmarks/dt_resolution_20260929/test/manifest.json'
        test_rows = read(test_manifest)['cases']
        model = VariableDrive(chosen['model_state'], device, dt)
        pp = {k: float(chosen['model_state'][k]) for k in ('v_th', 'reset_amp', 'tau_reset')}
        for r in test_rows:
            inp = np.load(r['input_file'])
            ids, raw = inp['ids'], inp['x']
            assert hashlib.sha256(raw.tobytes()).hexdigest() == r['input_sha256']
            n = round(r['duration_ms']/dt) + 1
            dense = np.zeros((500, n), np.float32)
            dense[ids,::stride] = raw * np.where(ids[:, None] < 250, 1, -1)
            with torch.inference_mode():
                xf = torch.fft.rfft(torch.from_numpy(dense).to(device)[None], n=fft_length(n))
                drive = model(xf, n)[0].cpu().numpy()
            voltage, ev = reset(drive, pp['v_th'], pp['reset_amp'], pp['tau_reset']/dt)
            ev = ev*dt
            ref = np.load(root / 'benchmarks/dt_resolution_20260929/test/references' / f"{r['case_id']}_reference.npz")
            reference = np.interp(np.arange(n)*dt, ref['time_ms'], ref['voltage'])
            mm = metric(reference[::stride], voltage[::stride], ref['spike_times'], 1.)
            ee = match_events(ref['spike_times'], ev)
            records.append(dict(case_id=r['case_id'], duration_ms=r['duration_ms'], input_sha256=r['input_sha256'],
                                voltage=mm, events=ee, reference_events_ms=ref['spike_times'].tolist(), predicted_events_ms=ev.tolist()))
            (out / 'test_predictions').mkdir(exist_ok=True)
            np.savez_compressed(out / 'test_predictions' / f"{r['case_id']}.npz", voltage=voltage, drive=drive,
                                reset_events_ms=ev, reference_voltage=reference, time_ms=np.arange(n)*dt)
        assert sha(cp) == selection['checkpoint_sha256']
        summary = aggregate([r['voltage'] for r in records])
        summary['events'] = event_aggregate([r['events'] for r in records])
    else:
        summary = None
    write(out / 'complete.json', dict(finished=True, smoke=a.smoke, protocol=protocol,
          selected=winner, checkpoint=str(cp), checkpoint_sha256=sha(cp), locked_before_test_at=locked,
          summary=summary, records=records, updates=updates, elapsed_s=time.perf_counter()-start_time))
    print('COMPLETE', name, summary, flush=True)


if __name__ == '__main__':
    main()
