"""Freeze complete637 and a120-input stress design before observing outputs."""
from pathlib import Path
import json,hashlib,datetime,shutil
import numpy as np
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');W=R/'benchmarks/unfiltered_stress_20261016'
CP=R/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt'
SHA='737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    W.mkdir(exist_ok=True);assert not (W/'manifest.json').exists(),'Frozen manifest exists; never replace it'
    assert sha(CP)==SHA
    src=R/'benchmarks/spike_panel_20261009/confirmatory/manifest.json';old=read(src)
    small=read(R/'benchmarks/solver_main379_20261016/manifest.json');old_ids={r['case_id'] for r in small['cases']}
    reference=read(src.parent/'reference_complete.json')['records'];ref={x['case_id']:x for x in reference}
    rows=[]
    for x in old['cases']:
        q={**x,'cohort':'full637','original_main379':x['case_id'] in old_ids}
        rr=ref[x['case_id']];q['screening_reference_dt_ms']=.1;q['screening_reference_spikes']=rr['spike_count']
        q['screening_minimum_isi_ms']=rr['minimum_isi_ms'];rows.append(q)
    assert len(rows)==637 and sum(r['original_main379'] for r in rows)==379
    ids=np.arange(500,dtype=np.int32);T=8000;n=T+1
    stress=[]
    families=['tonic_strength','event_packets','shared_rate_modulation','synchronous_events','ei_balance']
    for family in families:
      for level in range(3):
       for replicate in range(8):
        k=len(stress);seed=1110160000+k;rng=np.random.default_rng(seed)
        er=np.full(n,50.);ir=np.full(n,25.);params={}
        if family=='tonic_strength':
            er[:]=[50.,75.,100.][level];params=dict(exc_hz=float(er[0]),inh_hz=25.)
        elif family=='event_packets':
            duty=[.1,.2,.4][level];period=200;width=round(period*duty)
            er[:]=0.;er[(np.arange(n)%period)<width]=50./duty
            params=dict(exc_time_average_hz=50.,inh_hz=25.,period_ms=period,packet_width_ms=width,within_packet_exc_hz=50./duty)
        elif family=='shared_rate_modulation':
            amplitude=[.3,.6,.9][level];blocks=np.r_[np.ones(200),-np.ones(200)];rng.shuffle(blocks)
            er[:T]=50.*(1.+amplitude*np.repeat(blocks,20));er[T]=er[T-1]
            params=dict(exc_time_average_hz=50.,inh_hz=25.,common_modulation_amplitude=amplitude,rate_block_ms=20)
        elif family=='synchronous_events':
            params=dict(exc_hz=50.,inh_hz=25.,shared_excitatory_fraction=[.1,.3,.5][level],
                        encoding='shared Bernoulli event OR independent conditional Bernoulli; marginal p exactly0.05 per excitatory site')
        elif family=='ei_balance':
            er[:]=[60.,75.,90.][level];ir[:]=100.-er[0]
            params=dict(exc_hz=float(er[0]),inh_hz=float(ir[0]),sum_ei_rates_hz=100.)
        if family=='synchronous_events':
            frac=params['shared_excitatory_fraction'];p=.05;shared_p=p*frac
            shared=rng.random(n)<shared_p;conditional=(p-shared_p)/(1.-shared_p)
            ex=shared[None,:] | (rng.random((250,n))<conditional)
        else:ex=rng.random((250,n))<er[None,:]/1000.
        inh=rng.random((250,n))<ir[None,:]/1000.
        x=np.vstack([ex,inh]).astype(np.uint8);cid=f'stress_{k:04d}'
        path=W/'inputs'/f'{cid}.npz';path.parent.mkdir(exist_ok=True);np.savez_compressed(path,ids=ids,x=x)
        stress.append(dict(case_id=cid,cohort='stress120',original_main379=False,family=family,level=level,
              replicate=replicate,seed=seed,duration_s=8,duration_ms=T,A=500,site_mode='all',input_interval_ms=1.,
              input_file=str(path),input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
              ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),input_events=int(x.sum()),
              exc_input_events=int(ex.sum()),inh_input_events=int(inh.sum()),parameters=params))
    train=read(CP.parent/'training_manifest.json')['cases'];training_hashes={(x['input_sha256'],x['ids_sha256']) for x in train}
    combined=rows+stress;assert len(combined)==757
    assert all((x['input_sha256'],x['ids_sha256']) not in training_hashes for x in combined)
    assert len({(x['input_sha256'],x['ids_sha256'],x['duration_ms']) for x in combined})==757
    method_folders={'tcn':'tcn_author_default','elm':'elm_author_default','lstm':'lstm_common150'}
    methods={}
    for k,f in method_folders.items():
        p=R/'benchmarks/spike_panel_20261009/methods'/f/'protocol.json';j=read(p)
        methods[k]=dict(folder=str(p.parent),checkpoint=j['checkpoint'],checkpoint_sha256=j['checkpoint_sha256'],protocol_sha256=sha(p))
    protocol=dict(created=datetime.datetime.now().isoformat(),models_frozen_before_new_outputs=True,
        dbnn_checkpoint=str(CP),dbnn_checkpoint_sha256=SHA,baseline_models=methods,reference_dt_ms=.0125,
        solver_steps_ms=[.5,.25],learned_interval_ms=1.,event_tolerance_ms=5.,event_boundary_eps_ms=1e-6,
        voltage_mask_ms=10.,input_output_screening='None. All637 original candidates and all120 fixed stress inputs retained regardless of output.',
        full637_original_screening=dict(initial_reference_dt_ms=.1,total=637,retained_original_main=379,
            excluded_original_main=258,above3_spikes=258,isi_below50ms=1,both=1),
        stress_design=dict(cases=120,sites=500,duration_s=8,families=families,levels_per_family=3,seeds_per_level=8,
            control='Packet, shared-rate and synchronous families have nominal marginal E50Hz/I25Hz; tonic level0 is the independent control. Actual event totals are retained.'),
        split_validation=dict(training_and_validation_inputs=149,all757_input_hashes_disjoint=True),
        measurement='Prepared compute,5 measured repetitions following3 warmups for learned methods;3 repeats after1 warmup for detailed solvers. Timed learned GPU methods run serially on GPU0.',
        model_changes='None: weights, filters, reset, thresholds, calibration, learned capacities and training unchanged',
        manuscript_changes='None; standalone experimental report first')
    write(W/'protocol.json',protocol);write(W/'manifest.json',dict(created=protocol['created'],cases=combined))
    protected=[CP,CP.parent/'protocol.json',CP.parent/'training_manifest.json',src]
    protected += [Path(v['checkpoint']) for v in methods.values()]
    write(W/'protected_files.json',{str(p):sha(p) for p in protected})
    # Only existing fine references and corresponding-step solver outputs are
    # reused. Links point to immutable prior artifacts; they are never rewritten.
    for dt in [.0125,.5,.25]:
        dest=W/'outputs/reference1'/f'dt{dt:g}';dest.mkdir(parents=True,exist_ok=True)
        source=R/'benchmarks/solver_main379_20261016/outputs/reference1'/f'dt{dt:g}'
        for cid in old_ids:
            for suffix in ['.json','.npz']:(dest/(cid+suffix)).symlink_to(source/(cid+suffix))
    write(W/'original_reference_provenance.json',dict(source_manifest=str(src),source_manifest_sha256=sha(src),
         reference_rows_reused_per_step=379,source_work=str(R/'benchmarks/solver_main379_20261016')))
    print('FROZEN637+120 inputs; no output-conditioned screening. Manifest SHA256',sha(W/'manifest.json'),flush=True)
if __name__=='__main__':main()
