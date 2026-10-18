"""Lock every input before training/reference evaluation; never filter by output."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import itertools
import numpy as np
from common8 import read, write, sha

CELLS = ['granule', 'golgi', 'basket', 'hay_l5', 'human_l23', 'ca1']
PROFILES = ['ampa', 'ampa_nmda']


def project_for(cell):
    suffix = 'multineuron' if cell in CELLS[:3] else 'noncerebellar'
    return Path('/home/lcm/DBNN_'+suffix+'_20261002'), Path('/data/lcm/DBNN_'+suffix+'_20261002')


def positions_for(A, mode, seed, positions):
    rng = np.random.default_rng(seed)
    if A == 500:
        return np.arange(500, dtype=np.int32)
    n = A//2
    if mode == 'random':
        e, i = rng.choice(250, n, False), rng.choice(250, n, False)+250
    elif mode == 'stratified':
        e = np.round(np.linspace(0,249,n)).astype(int)
        i = e+250
    else:
        dist = np.array([p['distance_um'] for p in positions])
        e, i = np.argsort(dist[:250]), np.argsort(dist[250:])+250
        if mode == 'distal':
            e, i = e[::-1], i[::-1]
        e, i = e[:n], i[:n]
    return np.sort(np.r_[e,i]).astype(np.int32)


def input_case(cell_index, index, duration, A, hz, family, mode, positions):
    seed = 202610170000 + cell_index*10000 + index
    ids = positions_for(A,mode,seed,positions)
    rng = np.random.default_rng(seed+100000000)
    rates = np.full((A,duration+1), float(hz))
    if family == 'excitatory_gates':
        # Three 50-ms excitatory gates, over ongoing E/I background.
        for fraction in [.2,.5,.8]:
            start = round(duration*fraction)
            rates[ids<250,start:min(start+50,duration)] += min(6*hz,120)
    elif family == 'rate_changes':
        for a,b,factor in [(0,.25,.5),(.25,.5,2.),(.5,.75,.5),(.75,1.,2.)]:
            rates[:,round(duration*a):round(duration*b)] *= factor
    elif family == 'zero':
        rates[:] = 0
    raw = rng.poisson(rates/1000.)
    assert raw.max() < 128
    raw = raw.astype(np.uint8)
    raw[:,0] = raw[:,-1] = 0
    return ids, raw, dict(seed=seed,duration_ms=duration,A=A,rate_hz=hz,family=family,
                         site_mode=mode,input_interval_ms=1.,split='test',
                         input_events=int(raw.sum()),event_multiplicity='Native Poisson event counts; no gain scaling',
                         input_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                         ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);a=ap.parse_args()
    work=a.work.resolve();work.mkdir(parents=True,exist_ok=True)
    if (work/'locked_protocol.json').exists():
        d=read(work/'locked_protocol.json')
        for j in d['jobs']:
            assert sha(j['training_manifest'])==j['training_manifest_sha256']
            assert sha(j['test_manifest'])==j['test_manifest_sha256']
        print('MANIFESTS_ALREADY_LOCKED',len(d['jobs']),flush=True);return
    jobs=[]
    for ci,cell in enumerate(CELLS):
        project,data=project_for(cell)
        sites=project/'results'/cell/'screen/sites.json'
        positions=read(sites)['sites'];assert len(positions)==500
        for profile in PROFILES:
            base=work/'cells'/cell/profile;base.mkdir(parents=True,exist_ok=True)
            source=data/cell/profile/'train_1.0ms'
            # The unfiltered source manifest precedes all old spike-count/prefix selection.
            manifest=source/'candidate_manifest.json'
            if not manifest.exists():manifest=source/'manifest.json'
            rows=read(manifest)['cases'];assert len(rows)==96
            seen=set();hashes=set();train=[]
            for r in rows:
                assert r['split'] in ('fit','validation') and r['duration_ms']==6000
                assert r['input_interval_ms']==1. and r['seed'] not in seen
                seen.add(r['seed'])
                inp=np.load(r['input_file']);assert hashlib.sha256(inp['x'].tobytes()).hexdigest()==r['input_sha256']
                assert inp['x'].max()<128
                ref=np.load(r['reference_file']);assert np.isfinite(ref['voltage']).all()
                hashes.add(r['input_sha256'])
                train.append({**r,'reference_sha256':sha(r['reference_file']),
                              'reference_events':int(len(ref['spike_times']))})
            assert sum(r['split']=='fit' for r in train)==72
            assert sum(r['split']=='validation' for r in train)==24
            write(base/'training_manifest.json',dict(cases=train,source_manifest=str(manifest),
                source_manifest_sha256=sha(manifest),output_filtering=False,prefix_truncation=False))
            condition=read(source/'reference_condition.json')
            write(base/'reference_condition.json',condition)
            write(base/'sites.json',read(sites))
            (base/'test/inputs').mkdir(parents=True,exist_ok=True)
            (base/'test/references').mkdir(parents=True,exist_ok=True)
            tests=[]
            for ti,duration in enumerate([1000,2000,4000,8000,16000]):
                for ai,A in enumerate([64,128,256,500]):
                    for hi,hz in enumerate([10,25,50]):
                        for fi,family in enumerate(['stationary','excitatory_gates','rate_changes']):
                            mode=['random','stratified','proximal','distal'][(ti+ai+hi+fi)%4]
                            ids,raw,row=input_case(ci,len(tests),duration,A,hz,family,mode,positions)
                            assert row['input_sha256'] not in hashes and row['seed'] not in seen
                            cid=f'new_{cell}_{len(tests):04}'
                            path=base/'test/inputs'/f'{cid}.npz';np.savez_compressed(path,ids=ids,x=raw)
                            tests.append(dict(case_id=cid,input_file=str(path),
                                reference_file=str(base/'test/references'/f'{cid}.npz'),**row))
            for duration in [1000,16000]:
                ids,raw,row=input_case(ci,len(tests),duration,500,0,'zero','stratified',positions)
                cid=f'new_{cell}_{len(tests):04}';path=base/'test/inputs'/f'{cid}.npz'
                np.savez_compressed(path,ids=ids,x=raw)
                tests.append(dict(case_id=cid,input_file=str(path),
                                  reference_file=str(base/'test/references'/f'{cid}.npz'),**row))
            assert len(tests)==182 and len({r['seed'] for r in tests})==182
            write(base/'test/manifest.json',dict(cases=tests,output_filtering=False,
                locked_at=datetime.now(timezone.utc).isoformat(),
                design='5 durations x 4 active-site counts x 3 rates x 3 families + 2 zero controls; balanced cyclic assignment of four spatial modes'))
            jobs.append(dict(cell=cell,profile=profile,base=str(base),project=str(project),
                training_manifest=str(base/'training_manifest.json'),
                training_manifest_sha256=sha(base/'training_manifest.json'),
                test_manifest=str(base/'test/manifest.json'),test_manifest_sha256=sha(base/'test/manifest.json'),
                reference_condition_sha256=sha(base/'reference_condition.json'),sites_sha256=sha(base/'sites.json'),
                split_counts={s:sum(r['split']==s for r in train) for s in ['fit','validation']},
                fit_reference_events=sum(r['reference_events'] for r in train if r['split']=='fit'),
                validation_reference_events=sum(r['reference_events'] for r in train if r['split']=='validation')))
    protocol=dict(locked_at=datetime.now(timezone.utc).isoformat(),jobs=jobs,
        scope='All 12 cell/synapse conditions; random-init full-matrix DBNN, one kernel per site; primary L5PC unchanged',
        training=dict(epochs=2000,refine_epochs=400,lr=.003,seed=291049,window_ms=500,warmup_ms=150,
            batch_size=8,source_duration_ms=6000,validation_only_selection=True,
            threshold_grid='Fixed reference operating-point rest + 0.025, 0.5,1,...,24mV plus stage-current threshold; local offsets +/-0.375mV by0.125',
            reset_amplitudes_mv=[4,6,8,10,12],reset_tau_ms=[8,16,24,40],
            loss='Main voltage/derivative loss and validation criterion; no added event-training loss'),
        test=dict(total_conditions=12,inputs_per_condition=182,inputs_total=2184,
            seeds='202610170000 + cell_index*10000 + index; paired across AMPA/NMDA',
            outputs_used_for_selection=False,all_inputs_retained=True,bootstraps=10000,
            reference='Same fixed per-cell reference step and physical operating point as archived training data',
            interval_ms=1.,voltage_mask_ms=10.,event_tolerance_ms=5.,boundary_eps_ms=1e-6),
        manuscript_crosscell_update='Wait for complete results and user review; do not automatically replace paper tables')
    write(work/'locked_protocol.json',protocol)
    print('LOCKED',len(jobs),'JOBS',len(jobs)*182,'TEST_INPUTS',flush=True)


if __name__=='__main__':main()
