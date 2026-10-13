#!/usr/bin/env python3
"""Inspect the actual cell and re-evaluate the frozen checkpoint on recreated data.

This audit changes no training, benchmark inputs, or benchmark timing results.
The 10 held-out rasters/targets were sampled in the 2026-09-28 reproduction;
they are not claimed to be the checkpoint author's original dataset files.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(ref, pred, mask):
    r = np.asarray(ref, float)[mask]
    p = np.asarray(pred, float)[mask]
    e = p - r
    return dict(n=len(r), rmse_mv=float(np.sqrt(np.mean(e * e))),
                bias_mv=float(e.mean()), reference_variance=float(r.var()),
                error_variance=float(e.var()),
                ve=float(1-e.var()/r.var()), r2=float(1-np.mean(e*e)/r.var()),
                centered_error_ss=float(np.sum((e-e.mean())**2)),
                reference_ss=float(np.sum((r-r.mean())**2)),
                squared_error_ss=float(np.sum(e*e)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    root, out = a.root.resolve(), a.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(root/'scripts'), str(root/'source')]
    import sample_dbnn_data_repro as sampler
    from neuron import h
    cfg = dict(root=str(root/'source/ShaiEtAl2015'), biophys_file='L5PCbiophys5.hoc',
               morphology_file='cell1.asc', num_synapses=500, duration_ms=6000.,
               warmup_ms=500., dt=.1, sample_interval_ms=1., input_rate_hz=50.,
               exc_weight=1.88e-4, inh_weight=8e-5, exc_tau1_ms=.5, exc_tau2_ms=5.,
               inh_tau1_ms=1., inh_tau2_ms=10., inh_reversal_mv=-80., v_init=-75.,
               spike_threshold=-20., seed=20260425)
    state = sampler.build_worker_network(cfg)
    for ns in state['netstims']:
        ns.number = 0
    for idx, nc in enumerate(state['netcons']):
        nc.weight[0] = cfg['exc_weight'] if idx < 250 else cfg['inh_weight']
    h.finitialize(-75.)
    h.fcurrent()
    cell = state['cell']
    sections = {}
    for group in ['somatic', 'apical', 'basal', 'axonal']:
        ss = list(getattr(cell, group))
        dd = [s.psection()['density_mechs'] for s in ss]
        conductances = {}
        for d in dd:
            for mech, pars in d.items():
                for k, vs in pars.items():
                    if k.startswith('g') and ('bar' in k or mech == 'pas'):
                        conductances.setdefault(mech+'.'+k, []).extend(vs)
        sample = ss[0](.5)
        sections[group] = dict(section_count=len(ss), segment_count=sum(s.nseg for s in ss),
                               mechanisms=sorted(set(m for d in dd for m in d)),
                               conductance_range_s_per_cm2={k:[float(min(v)),float(max(v))] for k,v in conductances.items()},
                               example_section=ss[0].name(),
                               example_current_mA_per_cm2={k:float(getattr(sample,k)) for k in ['ina','ik'] if hasattr(sample,k)})
    synapses = []
    for idx in [0,249,250,499]:
        s = state['synapses'][idx]
        synapses.append(dict(site_id=idx, mechanism=s.hname(), tau1_ms=s.tau1,
                             tau2_ms=s.tau2, reversal_mV=s.e,
                             weight_uS=state['netcons'][idx].weight[0]))
    data_path = root/'data/shai_ei_500syn_6s_dataset.npz'
    d = np.load(data_path, allow_pickle=True)
    files = ['source/dbnn_checkpoint.pt','source/DBNN.py',
             'source/simulate_L5PC.py','source/simulate_time_compare.py',
             'source/ShaiEtAl2015/models/L5PCbiophys5.hoc',
             'source/ShaiEtAl2015/models/L5PCtemplate.hoc',
             'source/ShaiEtAl2015/morphologies/cell1.asc']
    files += [str(p.relative_to(root)) for p in (root/'source/ShaiEtAl2015/mod').glob('*.mod')]
    audit = dict(created=datetime.now().isoformat(), config=cfg,
                 runtime=dict(neuron_version=h.nrnversion(), celsius=h.celsius,
                              sections=sections, synapses=synapses),
                 recreated_dataset_sites_match=dict(
                     section_names=bool(np.array_equal(state['channel_section'], d['channel_section'])),
                     positions=bool(np.array_equal(state['channel_location'],d['channel_location'])),
                     channel_types=bool(np.array_equal(state['channel_type'],d['channel_type']))),
                 file_hashes={f:sha(root/f) for f in files})
    import torch
    from DBNN import DBNN_R
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ck = torch.load(root/'source/dbnn_checkpoint.pt', map_location='cpu', weights_only=False)
    model = DBNN_R(500,6001,torch.device('cpu'))
    model.load_state_dict(ck['model_state'],strict=True)
    model.eval()
    audit['checkpoint_config'] = {k:str(v) if isinstance(v,Path) else v for k,v in ck['config'].items()}
    audit['checkpoint_params'] = {k:ck['model_state'][k].tolist() for k in ['v_th','reset_amp','tau_reset','bilinear.bias']}
    signs = np.where(d['channel_type'].astype(str)=='exc',1.,-1.).astype(np.float32)
    rows, predictions = [], []
    t = np.arange(6001)
    # One legacy training-raster diagnostic plus all 10 recreated held-out rasters.
    for split, indices in [('train',[0]), ('test',list(range(10)))]:
        ip, op = d[split+'_ip'], d[split+'_op']
        for idx in indices:
            x = torch.from_numpy(ip[idx:idx+1].astype(np.float32)*signs[None,:,None])
            with torch.inference_mode():
                pred = model(x).numpy().ravel().copy()
            ref = op[idx].astype(float)
            spike_times = d[split+'_spike_time'][idx]
            mask = np.ones(6001,bool)
            for st in spike_times:
                mask &= ~((t>=st-5)&(t<=st+20))
            masks = dict(full=np.ones(6001,bool),no_spike_5_20=mask,
                         after100_no_spike_5_20=mask&(t>=100))
            rows.append(dict(split=split,index=idx,reference_spikes_ms=spike_times,
                             metrics={k:metrics(ref,pred,m) for k,m in masks.items()}))
            predictions.append(pred)
    audit['evaluation'] = dict(scope='Frozen original checkpoint; 10 recreated test samples, A=500, 50 Hz, 6 s; original continuous-event targets retained.',
                               ve='1-Var(pred-ref)/Var(ref)',
                               rows=rows, test_summary={})
    for mode in rows[0]['metrics']:
        ms = [r['metrics'][mode] for r in rows if r['split']=='test']
        audit['evaluation']['test_summary'][mode] = dict(
            mean_case_ve=float(np.mean([m['ve'] for m in ms])),
            mean_case_r2=float(np.mean([m['r2'] for m in ms])),
            pooled_rmse_mv=float(np.sqrt(sum(m['squared_error_ss'] for m in ms)/sum(m['n'] for m in ms))),
            variance_weighted_ve=float(1-sum(m['centered_error_ss'] for m in ms)/sum(m['reference_ss'] for m in ms)))
    out.write_text(json.dumps(audit,indent=2)+'\n')
    np.savez_compressed(out.with_suffix('.npz'), predictions=np.stack(predictions))
    print(json.dumps(dict(runtime=audit['runtime'],sites=audit['recreated_dataset_sites_match'],
                          test_summary=audit['evaluation']['test_summary'],output=str(out)),indent=2))


if __name__ == '__main__':
    main()
