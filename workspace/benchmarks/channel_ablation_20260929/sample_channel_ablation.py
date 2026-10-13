#!/usr/bin/env python3
"""Matched L5PC channel ablations, preserving morphology, passive membrane and input.

The existing full-active 100+10 sample dataset is the control. Each ablated
profile replays its continuous-event RNG streams and checks every stored raster.
"""
import argparse
from datetime import datetime
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np

STATE = {}


def init(root, profile, outdir):
    root = Path(root)
    sys.path.insert(0, str(root/'scripts'))
    import sample_dbnn_data_repro as sample
    cfg = dict(root=str(root/'source/ShaiEtAl2015'), biophys_file='L5PCbiophys5.hoc',
               morphology_file='cell1.asc', num_synapses=500, duration_ms=6000.,
               warmup_ms=500., dt=.1, sample_interval_ms=1., input_rate_hz=50.,
               exc_weight=1.88e-4, inh_weight=8e-5, exc_tau1_ms=.5, exc_tau2_ms=5.,
               inh_tau1_ms=1., inh_tau2_ms=10., inh_reversal_mv=-80., v_init=-75.,
               spike_threshold=-20., seed=20260425)
    sample.init_worker(cfg)
    cell = sample.WORKER_STATE['cell']
    secs = list(cell.dend)+list(cell.apic) if profile == 'passive_dendrites' else list(cell.all)
    removed = {}
    before = {}
    for sec in secs:
        before[sec.name()] = [(float(seg.pas.g),float(seg.pas.e),float(seg.cm)) for seg in sec]
        mechs = list(sec.psection()['density_mechs'])
        removed[sec.name()] = [m for m in mechs if m != 'pas']
        for mech in removed[sec.name()]:
            sec.uninsert(mech)
        assert before[sec.name()] == [(float(seg.pas.g),float(seg.pas.e),float(seg.cm)) for seg in sec]
        assert list(sec.psection()['density_mechs']) == ['pas']
    audit = dict(profile=profile,configuration=cfg,removed_mechanisms=removed,
                 remaining={group:sorted(set(m for sec in getattr(cell,group) for m in sec.psection()['density_mechs']))
                            for group in ['somatic','apical','basal','axonal']},
                 notes='Original pas conductance/reversal, cm, Ra, morphology, synapses and sites retained. No synthetic spikes/reset added to NEURON targets.')
    path = Path(outdir)/profile
    path.mkdir(parents=True,exist_ok=True)
    # Atomic write: all workers construct the same verified model.
    tmp=path/f'profile.{os.getpid()}.json';tmp.write_text(json.dumps(audit,indent=2)+'\n');tmp.replace(path/'profile.json')
    d=np.load(root/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True)
    STATE.update(sample=sample,path=path,train_ip=d['train_ip'],test_ip=d['test_ip'])


def run(idx):
    out=STATE['path']/f'sample_{idx:03d}.npz'
    if out.exists():
        z=np.load(out);return dict(index=idx,cached=True,spikes=int(z['spikes'].size))
    started=time.perf_counter()
    s=STATE['sample'].run_single_sample(idx)
    ref=STATE['train_ip'][idx] if idx<100 else STATE['test_ip'][idx-100]
    assert np.array_equal(s['ip'],ref), f'Input mismatch sample {idx}'
    np.savez_compressed(out,voltage=s['op'],spikes=np.array(s['spike_time'],float),input_sha256=hashlib.sha256(ref.tobytes()).hexdigest())
    return dict(index=idx,cached=False,spikes=s['spike_count'],wall_s=time.perf_counter()-started)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args();a.root=a.root.resolve();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
    source=np.load(a.root/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True)
    for profile in ['passive_dendrites','all_passive']:
        path=a.output/profile;path.mkdir(exist_ok=True)
        rows=[]
        with mp.get_context('spawn').Pool(a.workers,initializer=init,initargs=(str(a.root),profile,str(a.output))) as pool:
            for r in pool.imap_unordered(run,range(110),chunksize=1):
                rows.append(r)
                status=dict(profile=profile,complete=len(rows)==110,completed=len(rows),total=110,updated=datetime.now().isoformat(),rows=rows)
                (path/'status.json').write_text(json.dumps(status,indent=2)+'\n')
                if len(rows)%10==0 or len(rows)==110:print(profile,len(rows),'/ 110',flush=True)
        z=[np.load(path/f'sample_{i:03d}.npz') for i in range(110)]
        vs=np.stack([v['voltage'] for v in z]);ss=np.empty(110,object)
        for i,v in enumerate(z):ss[i]=v['spikes'].tolist()
        np.savez_compressed(path/'dataset.npz',train_op=vs[:100],test_op=vs[100:],train_spike_time=ss[:100],test_spike_time=ss[100:],
                            train_spike_count=np.array([len(s) for s in ss[:100]]),test_spike_count=np.array([len(s) for s in ss[100:]]),
                            channel_type=source['channel_type'],channel_section=source['channel_section'],channel_location=source['channel_location'])
        print(profile,'finished: mean spikes',np.mean([len(s) for s in ss]),flush=True)


if __name__=='__main__':main()
