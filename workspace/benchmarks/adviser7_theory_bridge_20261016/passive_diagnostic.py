"""Adviser 7: physical-morphology, fixed-feature DBNN error diagnostics.

This independent passive control never loads or writes the primary checkpoint.
Simulation and numerical fitting must run on the experiment server. No test
case is removed on the basis of voltage, spikes, or error. All test readouts
are frozen on fitting inputs. A fitted readout is not a test-set optimum.
"""
from pathlib import Path
import argparse
import fcntl
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
import platform
import sys
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

STATE = None
N = 501
NODES = np.array([.1, .2, .35, .5, .7, .85, 1.])
EPS = [.01, .02, .04, .08, .16, .32, .5, .64, 1., 2., 4., 8.]


def read(p):
    return json.loads(Path(p).read_text())


def write(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    q.replace(p)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def config(root):
    return dict(root=str(root / 'source/ShaiEtAl2015'),
                biophys_file='L5PCbiophys5.hoc', morphology_file='cell1.asc',
                num_synapses=500, duration_ms=500., warmup_ms=100., dt=.05,
                sample_interval_ms=1., input_rate_hz=20., exc_weight=1.88e-4,
                inh_weight=8e-5, exc_tau1_ms=.5, exc_tau2_ms=5.,
                inh_tau1_ms=1., inh_tau2_ms=10., inh_reversal_mv=-80.,
                v_init=-70., spike_threshold=-20., seed=20260425)


def init(root, dt):
    global STATE
    root = Path(root)
    sys.path.insert(0, str(root / 'scripts'))
    import sample_dbnn_data_repro as sampler
    from neuron import h, coreneuron
    cfg = config(root)
    net = sampler.build_worker_network(cfg)
    coreneuron.enable = False
    pc = h.ParallelContext(); pc.nthread(1); pc.thread_busywait(0)
    removed = {}; sections = []
    for sec in net['cell'].all:
        before = list(sec.psection()['density_mechs'])
        for mechanism in before:
            if mechanism != 'pas':
                sec.uninsert(mechanism)
                removed[mechanism] = removed.get(mechanism, 0) + 1
        assert 'pas' in sec.psection()['density_mechs'], sec.name()
        for seg in sec:
            seg.pas.e = -70.
        assert set(sec.psection()['density_mechs']) == {'pas'}
        sections.append(dict(name=sec.name(), nseg=int(sec.nseg), L_um=float(sec.L),
                             Ra_ohm_cm=float(sec.Ra),
                             cm_uf_cm2=[float(seg.cm) for seg in sec],
                             g_pas_s_cm2=[float(seg.pas.g) for seg in sec]))
    for ns in net['netstims']:
        ns.number = 0
    for nc in net['netcons']:
        nc.weight[0] = 0
    h.distance(0, .5, sec=net['cell'].soma[0])
    dist = np.array([h.distance(s.get_segment().x, sec=s.get_segment().sec)
                     for s in net['synapses']])
    ids = []
    for candidates in [np.arange(250), np.arange(250, 500)]:
        order = candidates[np.argsort(dist[candidates], kind='stable')]
        ids += order[np.linspace(0, 249, 8).round().astype(int)].tolist()
    ids = np.array(ids)
    direct = [h.NetCon(None, net['synapses'][int(i)]) for i in ids]
    for nc in direct:
        nc.delay = 0
    STATE = dict(h=h, net=net, direct=direct, pc=pc, dt=dt, ids=ids,
                 metadata=dict(candidate_sites=500, selected_sites=16,
                               site_ids=ids.tolist(), distance_um=dist[ids].tolist(),
                               section_names=net['channel_section'][ids].tolist(),
                               section_locations=net['channel_location'][ids].tolist(),
                               removed_density_mechanisms=removed, sections=sections,
                               resting_voltage_mv=-70., conversion='Remove all density mechanisms except original pas; keep morphology, nseg, Ra, cm, g_pas; set every e_pas=-70 mV. Exp2Syn unchanged.',
                               synapses={k:v for k,v in cfg.items() if 'tau' in k or 'weight' in k or 'reversal' in k}))


def simulate(job):
    name, x, eps = job
    s = STATE; h = s['h']
    for j, nc in enumerate(s['direct']):
        nc.weight[0] = eps * (1.88e-4 if j < 8 else 8e-5)
    v = h.Vector().record(s['net']['cell'].soma[0](.5)._ref_v, 1.)
    t = h.Vector().record(h._ref_t, 1.)
    h.dt = s['dt']; h.steps_per_ms = 1 / h.dt; h.secondorder = 0
    h.cvode_active(0); h.finitialize(-70.); h.fcurrent(); h.frecord_init()
    # Passive equilibrium is exactly uniform. A 100 ms no-input history is
    # still supplied, with no synaptic or reset state at the recording start.
    for j, nc in enumerate(s['direct']):
        for onset in np.flatnonzero(x[j]):
            nc.event(100. + float(onset))
    h.continuerun(600.)
    tt = np.array(t.as_numpy(), copy=True)
    vv = np.array(v.as_numpy(), copy=True)
    out = np.interp(100. + np.arange(N), tt, vv)
    assert np.isfinite(out).all() and len(out) == N
    return name, eps, out


def profile(name, family, split, x):
    return dict(name=name, family=family, split=split,
                event_indices=[np.flatnonzero(row).tolist() for row in x],
                input_sha256=hashlib.sha256(x.tobytes()).hexdigest())


def make_profiles():
    rows = []
    for j in range(64):
        rng = np.random.default_rng(7101600 + j)
        x = np.zeros((16, N), np.uint8)
        family = ['synchronous', 'asynchronous', 'stationary', 'packets'][j // 16]
        for site in range(16):
            if family == 'synchronous':
                times = [20 + j*3 + (0 if site < 8 else (j % 4) * 10)]
            elif family == 'asynchronous':
                times = rng.integers(20, 301, size=2)
            elif family == 'stationary':
                times = np.flatnonzero(rng.random(380) < .02) + 20
            else:
                start = rng.integers(20, 201)
                times = start + np.arange(4) * [2, 5, 10, 15][j % 4]
            # Vary subsets by declared RNG, independently of model outputs.
            if j < 8 or rng.random() > .25:
                x[site, np.asarray(times, int)] = 1
        rows.append(profile(f'fit_{j:03}', family, 'fit', x))
    for j in range(32):
        rng = np.random.default_rng(7201600 + j)
        x = np.zeros((16, N), np.uint8)
        family = 'unseen_delay' if j < 8 else ('unseen_spacing' if j < 16 else 'independent_random')
        for site in range(16):
            if family == 'unseen_delay':
                times = [30 + (0 if site < 8 else [7, 23, 47, 83][j % 4])]
            elif family == 'unseen_spacing':
                times = 30 + np.arange(5) * [26, 37][j % 2]
            else:
                times = np.flatnonzero(rng.random(380) < [.01, .035][j % 2]) + 20
            if j % 2 == 0 or rng.random() > .25:
                x[site, np.asarray(times, int)] = 1
        rows.append(profile(f'test_{j:03}', family, 'test', x))
    for j in range(16):
        x = np.zeros((16, N), np.uint8); x[j, 20] = 1
        rows.append(profile(f'isolated_{j:03}', 'isolated_impulse', 'kernel_fit', x))
    hashes = [r['input_sha256'] for r in rows]
    # Synchronous fit repetitions can coincide only in the deliberately
    # declared fit family. Cross-split identity is forbidden.
    assert not ({r['input_sha256'] for r in rows if r['split'] == 'fit'} &
                {r['input_sha256'] for r in rows if r['split'] == 'test'})
    return rows


def raster(row):
    x = np.zeros((16, N), np.uint8)
    for j, indices in enumerate(row['event_indices']):
        x[j, indices] = 1
    return x


def strength_tag(eps):
    return format(eps, '.8g').replace('.', 'p')


def cache_path(out, dt, name, eps):
    return out / 'raw' / f'dt{dt:g}' / f'{name}_e{strength_tag(eps)}.npz'


def collect(root, out, rows, levels, dt, workers):
    jobs = []
    for row in rows:
        for eps in levels:
            p = cache_path(out, dt, row['name'], eps)
            if not p.exists():
                jobs.append((row['name'], raster(row), float(eps)))
    if not jobs:
        return
    folder = out / 'raw' / f'dt{dt:g}'; folder.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                             initializer=init, initargs=(str(root), dt)) as pool:
        futures = [pool.submit(simulate, job) for job in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            name, eps, v = f.result()
            np.savez_compressed(cache_path(out, dt, name, eps), voltage_mv=v,
                                epsilon=eps, dt_ms=dt)
            if n == 1 or n % 20 == 0 or n == len(jobs):
                status = dict(stage='simulation', solver_dt_ms=dt,
                              current_stage_completed=n, current_stage_total=len(jobs),
                              stage_elapsed_s=time.monotonic()-start)
                write(out / 'status.json', status); print(status, flush=True)


def load(out, dt, row, eps):
    return np.load(cache_path(out, dt, row['name'], float(eps)))['voltage_mv'] + 70.


def coefficients(out, dt, row, scale=.08, degree=5):
    responses = np.array([load(out, dt, row, scale*q) for q in NODES])
    design = np.stack([NODES**j for j in range(1, degree+1)], 1)
    c = np.linalg.lstsq(design, responses, rcond=None)[0]
    return c[0]/scale, c[1]/scale**2, float(np.max(abs(design@c-responses)))


def fit_kernel(v1):
    age = np.arange(N) - 20.; valid = age >= 0
    age = np.maximum(age, 0.)
    def build(logtau):
        tr, td = np.exp(logtau)
        k = (1 - np.exp(-age/tr)) * np.exp(-age/td) * valid
        amp = np.dot(v1, k) / max(np.dot(k, k), 1e-300)
        return amp*k, float(amp)
    fits = []
    for initial in [(1., 10.), (5., 20.), (10., 60.)]:
        sol = least_squares(lambda p:(build(p)[0]-v1)/max(rms(v1), 1e-12),
                            np.log(initial), bounds=(np.log([.05, .1]), np.log([500., 1000.])),
                            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=2000)
        pred, amp = build(sol.x)
        fits.append((rms(pred-v1), sol, pred, amp))
    error, sol, pred, amp = min(fits, key=lambda r:r[0])
    tr, td = np.exp(sol.x)
    return dict(tau_rise_ms=float(tr), tau_decay_ms=float(td),
                physical_impulse_amplitude_mv=amp, residual_rms_mv=error,
                target_rms_mv=rms(v1), relative_rms=error/max(rms(v1), 1e-300)), pred


def features(row, kernels):
    x = raster(row)
    t = np.arange(N)
    ks = np.array([k['physical_impulse_amplitude_mv'] * (1-np.exp(-t/k['tau_rise_ms'])) *
                   np.exp(-t/k['tau_decay_ms']) for k in kernels])
    return fftconvolve(x, ks, mode='full', axes=-1)[:, :N].T


def products(y):
    i, j = np.triu_indices(y.shape[1])
    return y[:, i] * y[:, j]


def projection(x, target):
    scale = np.sqrt(np.mean(x*x, axis=0))
    active = scale > max(float(scale.max())*1e-12, 1e-30)
    z = x[:, active] / scale[active]
    c, _, rank, sv = np.linalg.lstsq(z, target, rcond=1e-10)
    w = np.zeros(x.shape[1]); w[active] = c/scale[active]
    residual = target-x@w
    normal = rms(z.T@residual / len(target))
    return w, dict(rank=int(rank), features=int(x.shape[1]), active_columns=int(active.sum()),
                   training_rmse_mv=rms(residual), normal_equation_residual=normal,
                   retained_singular_ratio=float(sv[int(rank)-1]/sv[0]),
                   rcond=1e-10), scale


def fit_gradient_readout(x, target, scale):
    """Independent conditional training, not a retrain of the main network.

    Fixed physical kernels; ordinary finite-strength voltage MSE. The target
    and sample weights are exactly those of the conditional LS comparison.
    AdamW is run with a declared schedule, with no held-out selection.
    """
    import torch
    torch.set_num_threads(1)
    z = x / np.maximum(scale, 1e-30)
    # Exact full-batch MSE through its small Gram matrix; the objective and
    # gradient are identical to forming every sample prediction at each step.
    gram = torch.tensor(z.T@z/len(z), dtype=torch.float64)
    corr = torch.tensor(z.T@target/len(z), dtype=torch.float64)
    constant = float(np.mean(target**2))
    torch.manual_seed(71016)
    w = torch.nn.Parameter(torch.randn(x.shape[1], dtype=torch.float64)*1e-4)
    opt = torch.optim.AdamW([w], lr=.01, weight_decay=0.)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=3000, eta_min=1e-5)
    trace = []
    for step in range(3000):
        opt.zero_grad(set_to_none=True); loss = w@(gram@w)-2*w@corr+constant
        loss.backward(); torch.nn.utils.clip_grad_norm_([w], 1.); opt.step(); sched.step()
        if step % 100 == 0 or step == 2999:
            trace.append(dict(step=step+1, loss=float(loss.detach())))
    return w.detach().numpy()/np.maximum(scale, 1e-30), dict(
        optimizer='AdamW', updates=3000, initial_lr=.01, minimum_lr=1e-5,
        weight_decay=0., initialization_seed=71016, coordinates='column RMS response coordinates',
        objective='ordinary finite-strength voltage MSE, fixed kernels and baseline',
        model_scope='Independent 16-site passive diagnostic, not the primary training pipeline or checkpoint',
        loss_trace=trace)


def analyze(out, rows, metadata, smoke=False):
    isolated = [r for r in rows if r['split'] == 'kernel_fit']
    kernels = []; isolated_arrays = {}
    for j, row in enumerate(isolated):
        v1, v2, _ = coefficients(out, .05, row)
        fit, y = fit_kernel(v1); fit.update(site_index=j, site_id=metadata['site_ids'][j])
        fit['omega_signed_encoding_mv'] = fit['physical_impulse_amplitude_mv'] * (1 if j < 8 else -1)
        kernels.append(fit); isolated_arrays[row['name']+'_V1'] = v1
        isolated_arrays[row['name']+'_fit'] = y
    main = [r for r in rows if r['split'] in ('fit', 'test')]
    data = {}
    for r in main:
        v1, v2, fitgap = coefficients(out, .05, r)
        y = features(r, kernels)
        data[r['name']] = dict(v1=v1, v2=v2, y=y, phi=products(y), extraction_gap=fitgap)
    fitrows = [r for r in main if r['split'] == 'fit']
    testrows = [r for r in main if r['split'] == 'test']
    phi = np.concatenate([data[r['name']]['phi'] for r in fitrows])
    v2 = np.concatenate([data[r['name']]['v2'] for r in fitrows])
    w2, fit2, _ = projection(phi, v2)
    # Compare gradient-trained and optimal readouts with exactly the same
    # finite-strength objective, not with the distinct V2 projection target.
    xvol = []; tvol = []
    for r in fitrows:
        d = data[r['name']]
        for e in [.5, 1., 2.]:
            xvol.append(e*e*d['phi'])
            tvol.append(load(out, .05, r, e)-e*d['y'].sum(1))
    xvol = np.concatenate(xvol); tvol = np.concatenate(tvol)
    wv, fitv, scale = projection(xvol, tvol)
    wt, training = fit_gradient_readout(xvol, tvol, scale)
    orthogonality = dict(
        finite_voltage_optimal_mse=rms(xvol@wv-tvol)**2,
        finite_voltage_trained_mse=rms(xvol@wt-tvol)**2,
        prediction_gap_mse=rms(xvol@(wt-wv))**2,
        scope='Same fitting samples and same finite-voltage MSE only; no claim of test-set orthogonality or main-training convergence')
    orthogonality['identity_abs_gap'] = abs(orthogonality['finite_voltage_trained_mse']-
        orthogonality['finite_voltage_optimal_mse']-orthogonality['prediction_gap_mse'])
    records = []; raw = dict(isolated_arrays)
    for r in main:
        d = data[r['name']]; a = d['v1']-d['y'].sum(1)
        b = d['v2']-d['phi']@w2
        for e in EPS:
            true = load(out, .05, r, e)
            remainder = true-e*d['v1']-e*e*d['v2']
            pred = e*d['y'].sum(1)+e*e*d['phi']@w2
            actual_error = true-pred
            # Check the exact vector identity before taking norms.
            identity_gap = float(np.max(abs(actual_error-(e*a+e*e*b+remainder))))
            assert identity_gap < 1e-10
            budget = e*rms(a)+e*e*rms(b)+rms(remainder)
            assert rms(actual_error) <= budget+1e-10
            rec = dict(name=r['name'], split=r['split'], family=r['family'], epsilon=e,
                       delta1_mv=rms(a), delta2_fixed_mv=rms(b),
                       true_response_rms_mv=rms(true), true_response_peak_mv=float(np.max(abs(true))),
                       first_order_contribution_mv=e*rms(a), second_order_contribution_mv=e*e*rms(b),
                       higher_order_residual_mv=rms(remainder), total_rmse_mv=rms(actual_error),
                       triangle_budget_mv=budget, vector_identity_max_gap_mv=identity_gap,
                       finite_voltage_optimal_fixed_readout_rmse_mv=rms(true-e*d['y'].sum(1)-e*e*d['phi']@wv),
                       finite_voltage_trained_readout_rmse_mv=rms(true-e*d['y'].sum(1)-e*e*d['phi']@wt),
                       physical_first_rmse_mv=rms(true-e*d['v1']),
                       physical_second_rmse_mv=rms(remainder),
                       fitted_features_first_only_rmse_mv=rms(true-e*d['y'].sum(1)))
            records.append(rec)
        raw[r['name']+'_V1'] = d['v1']; raw[r['name']+'_V2'] = d['v2']
        raw[r['name']+'_features'] = d['y']; raw[r['name']+'_V2_projection'] = d['phi']@w2
    groups = []
    for split in ['fit', 'test']:
        for e in EPS:
            rr = [r for r in records if r['split'] == split and r['epsilon'] == e]
            numeric = [k for k in rr[0] if k not in ['name', 'family', 'split', 'epsilon']]
            groups.append(dict(split=split, epsilon=e, inputs=len(rr),
                               **{k:float(np.sqrt(np.mean([r[k]**2 for r in rr]))) for k in numeric}))
    controls = []
    selected = [r for r in main if r['name'] in ['fit_000','fit_017','fit_033','fit_049','test_000','test_008','test_016','test_017']]
    selected += isolated[::5]
    for r in selected:
        a1, a2, _ = coefficients(out, .05, r)
        for tag, dt, scale2, degree in [('half_extraction',.05,.04,5), ('degree4',.05,.08,4), ('half_step',.025,.08,5)]:
            b1, b2, gap = coefficients(out, dt, r, scale2, degree)
            controls.append(dict(name=r['name'], control=tag,
                                 V1_rms_gap_mv=rms(a1-b1), V2_rms_gap_mv=rms(a2-b2),
                                 V1_relative_rms_gap=rms(a1-b1)/max(rms(a1),1e-30),
                                 V2_relative_rms_gap=rms(a2-b2)/max(rms(a2),1e-30),
                                 polynomial_max_fit_gap_mv=gap))
    np.savez_compressed(out/'analysis_arrays.npz', **raw, physical_second_readout=w2,
                        finite_voltage_optimal_readout=wv, finite_voltage_trained_readout=wt)
    result = dict(finished=True, smoke=smoke, execution_host=platform.node(),
                  model_metadata=metadata, profiles=rows, kernels=kernels, controls=controls,
                  physical_second_projection_fit=fit2, finite_voltage_projection_fit=fitv,
                  conditional_training=training, fitting_set_orthogonality=orthogonality,
                  groups=groups, records=records, source_sha256=sha(__file__),
                  main_checkpoint_loaded_or_changed=False,
                  time_average='Equal weight over all 501 samples, including the no-input prefix; also retain raw traces for alternate windows',
                  interpretation=['delta1 need not vanish in a spatial cell with one double-exponential kernel per site.',
                                  'Second-order projection coefficients are frozen across strengths and all held-out timings.',
                                  'The fitting projection residual is minimal on fitting samples only.',
                                  'The AdamW-vs-LS comparison is a conditional, finite-strength passive readout diagnostic, not a claim about main active-model optimization.',
                                  'No test inputs are filtered by output amplitudes, spike count, or prediction quality.'])
    write(out/'complete.json', result)
    write(out/'status.json', dict(stage='complete', full_results=not smoke))
    print('PASSIVE_COMPLETE', len(records), flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--root', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path); ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--preflight', action='store_true'); args = ap.parse_args()
    assert platform.node() == 'ps', 'Run simulations on the authorized 5090 experiment host'
    root, out = args.root.resolve(), args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    guard = open(out/'run.lock', 'w')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    init(root, .05); metadata = STATE['metadata']
    # One zero and one E-only run verify a quiet passive background, actual
    # delivery, fixed site selection, and recording endpoints before dispatch.
    x = np.zeros((16, N), np.uint8)
    _, _, bg = simulate(('zero', x, 0.)); x[0,20] = 1
    _, _, response = simulate(('single', x, 1.))
    assert np.max(abs(bg+70)) < 1e-8
    assert np.max(response-bg) > 0
    write(out/'preflight.json', dict(passed=True, execution_host=platform.node(),
          background_max_deviation_mv=float(np.max(abs(bg+70))),
          first_site_epsp_peak_mv=float(np.max(response-bg)), metadata=metadata))
    if args.preflight:
        print('PREFLIGHT_PASSED', flush=True); return
    rows = make_profiles()
    protocol = dict(model=metadata, profiles=rows, strength_levels=EPS,
                    extraction_nodes=NODES.tolist(), extraction_scale=.08,
                    extraction_degree=5, main_solver_dt_ms=.05, control_solver_dt_ms=.025,
                    observations_ms=1., observation_samples=N, workers=args.workers,
                    checkpoint_policy='Main active DBNN never loaded or changed',
                    script_sha256=sha(__file__))
    if (out/'protocol.json').exists():
        assert read(out/'protocol.json') == protocol, 'Changed protocol in a resumable directory'
    write(out/'protocol.json', protocol)
    # Network instances used in spawned processes are separate from preflight.
    collect(root, out, rows, .08*NODES, .05, args.workers)
    mainrows = [r for r in rows if r['split'] in ('fit','test')]
    collect(root, out, mainrows, EPS, .05, args.workers)
    selected = [r for r in rows if r['name'] in ['fit_000','fit_017','fit_033','fit_049','test_000','test_008','test_016','test_017']]
    selected += [r for r in rows if r['split']=='kernel_fit'][::5]
    collect(root, out, selected, .04*NODES, .05, args.workers)
    collect(root, out, selected, list(.08*NODES)+[1.,4.], .025, args.workers)
    write(out/'status.json', dict(stage='fitting_and_decomposition'))
    analyze(out, rows, metadata)


if __name__ == '__main__':
    main()
