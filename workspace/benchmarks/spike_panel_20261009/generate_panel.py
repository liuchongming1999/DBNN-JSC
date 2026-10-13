"""Independent inputs; pilot physiology then a frozen confirmatory panel.

No DBNN output is used to generate or select inputs. Spatial groups use actual
soma path distance on the fixed 500-site geometry, not section coordinates.
"""
import argparse
import hashlib
from pathlib import Path
import sys
from datetime import datetime
import numpy as np
from common_panel import read,write,config,sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--phase',choices=['pilot','confirmatory'],required=True)
    a=p.parse_args();root=a.root.resolve();work=a.work.resolve()
    folder=work/a.phase;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'manifest.json').exists(): raise RuntimeError('Manifest already exists; preserve inputs')
    metadata=work/'site_metadata.json'
    if not metadata.exists():
        sys.path.insert(0,str(root/'scripts'))
        import sample_dbnn_data_repro as sample
        s=sample.build_worker_network(config(root));h=s['h'];cell=s['cell']
        h.distance(0,.5,sec=cell.soma[0])
        rows=[]
        for i,syn in enumerate(s['synapses']):
            seg=syn.get_segment()
            rows.append(dict(id=i,kind=str(s['channel_type'][i]),section=seg.sec.name(),section_position=float(seg.x),
                             soma_path_distance_um=float(h.distance(seg.x,sec=seg.sec))))
        write(metadata,dict(model=config(root),sites=rows))
    sites=read(metadata)['sites'];dist=np.array([r['soma_path_distance_um'] for r in sites])
    cases=[]

    def add(family,T,A,mode,rep,exc=50.,inh=50.,width=None,base=25.):
        idx=len(cases);seed=(810090000 if a.phase=='pilot' else 910090000)+idx
        rng=np.random.default_rng(seed)
        selected=[]
        for pool,count in [(np.arange(250),A//2),(np.arange(250,500),A-A//2)]:
            if mode=='all': chosen=pool
            elif mode=='random': chosen=rng.choice(pool,count,replace=False)
            elif mode=='stratified': chosen=pool[np.argsort(dist[pool])[np.linspace(0,len(pool)-1,count).round().astype(int)]]
            else:
                ordered=pool[np.argsort(dist[pool])]
                candidates=ordered[:max(count,125)] if mode=='proximal' else ordered[-max(count,125):]
                chosen=rng.choice(candidates,count,replace=False)
            selected.extend(chosen.tolist())
        ids=np.sort(np.array(selected,dtype=np.int32));n=T*1000+1;t=np.arange(n)
        er=np.full(n,exc,dtype=float);ir=np.full(n,inh,dtype=float);epochs=[]
        if family=='gated_excitation':
            er[:]=base;ir[:]=base
            centers=np.linspace(1000,T*1000-1000,3) if T>=4 else np.array([T*500])
            for center in centers:
                lo=int(center-width/2);hi=int(center+width/2)
                er[lo:hi]=exc;ir[lo:hi]=inh
                epochs.append(dict(start_ms=lo,end_ms=hi,exc_hz=exc,inh_hz=inh))
        elif family=='rate_steps':
            period=1000
            phases=(t//period)%2
            er=np.where(phases==0,25.,50.);ir=er.copy()
        elif family=='zero': er[:]=0;ir[:]=0
        rates=np.where(ids[:,None]<250,er[None],ir[None])
        x=(rng.random((A,n))<rates/1000).astype(np.uint8)
        cid=f'{a.phase}_{idx:04d}'
        path=folder/'inputs'/f'{cid}.npz';path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,ids=ids,x=x)
        row=dict(case_id=cid,panel=a.phase,duration_s=T,duration_ms=T*1000,A=A,site_mode=mode,
                 family=family,replicate=rep,seed=seed,input_interval_ms=1.,input_file=str(path),
                 input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                 input_events=int(x.sum()),rate_hz=float(exc),exc_hz=exc,inh_hz=inh,
                 background_hz=base if family=='gated_excitation' else None,gate_width_ms=width,gate_epochs=epochs,
                 soma_path_distance_range_um=[float(dist[ids].min()),float(dist[ids].max())])
        cases.append(row)

    if a.phase=='pilot':
        for rep in range(8): add('stationary',8,500,'all',rep)
        for width in [40,80,120,200]:
            for rep in range(4): add('gated_excitation',8,500,'all',rep,50.,10.,width)
        for rep in range(4): add('rate_steps',8,500,'all',rep)
    else:
        pilot=read(work/'pilot/reference_complete.json')['records']
        source=read(work/'pilot/manifest.json')['cases'];byid={r['case_id']:r for r in source}
        groups={}
        for r in pilot:
            inp=byid[r['case_id']]
            if inp['family']=='gated_excitation':
                groups.setdefault(inp['gate_width_ms'],[]).append(r)
        # Choose gate duration from NEURON only: prefer 1-3 well-separated spikes.
        ranking=[]
        for width,refs in groups.items():
            good=[r for r in refs if 1<=r['spike_count']<=3 and r['minimum_isi_ms']>=50]
            ranking.append((len(good),sum(r['spike_count'] for r in good),-width,width))
        gate_width=max(ranking)[-1]
        decision=dict(chosen_width_ms=gate_width,ranking=ranking,
                      criterion='NEURON-only pilot: maximize cases with 1-3 spikes and ISI>=50ms, then reference events, then shorter gate',
                      pilot_seed_disjoint=True,model_predictions_consulted=False)
        write(work/'pilot_design_decision.json',decision)
        # 512 independent natural-input draws, covering all five durations.
        for T,reps in [(1,32),(2,32),(4,64),(8,256),(16,128)]:
            for rep in range(reps): add('stationary',T,500,'all',rep)
        # 80 examples probe time-varying input and do not alter weights/channels.
        for T in [4,8,16]:
            for rep in range(16): add('gated_excitation',T,500,'all',rep,50.,10.,gate_width)
        for T in [2,4,8,16]:
            for rep in range(8): add('rate_steps',T,500,'all',rep)
        # Sparse/spatial/background controls retain false-positive assessment.
        for A in [64,128,256]:
            for mode in ['random','stratified','proximal','distal']:
                for rep in range(2): add('stationary',8,A,mode,rep,50.,50.)
        for rate in [10.,25.]:
            for rep in range(8): add('stationary',8,500,'all',rep,rate,rate)
        for T in [1,2,4,8,16]: add('zero',T,500,'all',0,0.,0.)
    keys=[(r['input_sha256'],r['ids_sha256'],r['duration_ms']) for r in cases]
    assert len(set(keys))==len(keys)
    checkpoint=root/'benchmarks/diagonal_tuning_1ms_20261009/full_relaxed/candidates/stage1_joint/checkpoint.pt'
    manifest=dict(created=datetime.now().isoformat(),phase=a.phase,neuron_config=config(root),site_metadata_sha256=sha(metadata),
                  checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),checkpoint_frozen=True,
                  minimum_target_reference_events=300,reference_dt_ms=.1,output_interval_ms=1.,
                  spike_analysis=dict(low_density='all reference trajectories with 0-3 events and minimum ISI>=50ms',
                                      all_generated_inputs_also_reported=True,event_tolerance_ms=5,voltage_mask_ms=10,
                                      bootstrap='Resample independent trajectories, not repeated prefixes or individual spikes'),
                  input_design='Independent 1-ms Bernoulli/Poisson rasters on fixed physical sites; no old prefix reuse; physiological channel and synapse weights unchanged',
                  model_definition='Full 500x500 quadratic W with trainable diagonal; no zero-diagonal trials',cases=cases)
    write(folder/'manifest.json',manifest)
    print('GENERATED',a.phase,len(cases),flush=True)


if __name__=='__main__':main()
