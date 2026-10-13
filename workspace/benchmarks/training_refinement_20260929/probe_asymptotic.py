"""Controlled conductance-amplitude probe; does not evaluate the main31 panel."""
import argparse, hashlib, inspect, json, sys
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--profile',choices=['full_active','all_passive'],required=True);a=p.parse_args();root=a.root.resolve()
    work=root/'benchmarks/training_refinement_20260929';out=work/'asymptotic'/a.profile;out.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(root/'scripts'),str(root/'benchmarks/neuron_single_expanded_20260929')]
    import sample_dbnn_data_repro as sample
    import run_reference
    from neuron import h,coreneuron
    cfg=dict(root=str(root/'source/ShaiEtAl2015'),biophys_file='L5PCbiophys5.hoc',morphology_file='cell1.asc',num_synapses=500,duration_ms=1000.,warmup_ms=500.,dt=.1,sample_interval_ms=1.,input_rate_hz=50.,exc_weight=1.88e-4,inh_weight=8e-5,exc_tau1_ms=.5,exc_tau2_ms=5.,inh_tau1_ms=1.,inh_tau2_ms=10.,inh_reversal_mv=-80.,v_init=-75.,spike_threshold=-20.,seed=20260425)
    state=sample.build_worker_network(cfg);coreneuron.enable=False
    if a.profile=='all_passive':
        for sec in state['cell'].all:
            for mech in list(sec.psection()['density_mechs']):
                if mech!='pas':sec.uninsert(mech)
            assert list(sec.psection()['density_mechs'])==['pas']
    pc=h.ParallelContext();pc.nthread(1);pc.thread_busywait(0)
    ncs=[h.NetCon(None,syn) for syn in state['synapses']]
    # Identical replay and observation protocol; only synaptic conductance amplitude changes.
    src=inspect.getsource(run_reference.run_case)
    src=src.replace('1.88e-4 if int(site) < 250 else 8.0e-5','row["weight_scale"] * (1.88e-4 if int(site) < 250 else 8.0e-5)')
    env=dict(run_reference.__dict__);exec(src,env);run=env['run_case']
    allrows=[]
    for A in [64,500]:
        inp=np.load(work/'data/inputs'/f'new_A{A}_hz50_r3.npz');ids=inp['ids'];x=inp['x'][:,:1001]
        ip=out/f'input_A{A}.npz';np.savez_compressed(ip,ids=ids,x=x)
        refs={};events={}
        for scale in [0.,.0625,.125,.25,.5,1.,2.]:
            cid=f'A{A}_scale{scale:g}';r=dict(case_id=cid,input_file=ip.name,A=A,duration_ms=1000.,site_mode='validation subset',rate_hz=50.,weight_scale=scale,input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),ids_sha256=hashlib.sha256(ids.astype(np.int32).tobytes()).hexdigest())
            q=run(h,state,ncs,r,out,out);arr=np.load(out/q['voltage_file']);refs[scale]=arr['voltage'].astype(float);events[scale]=arr['spike_times']
            print(a.profile,cid,q['spike_count'],flush=True)
        scales=np.array([.0625,.125]);D=np.stack([scales,scales**2],axis=1);y=np.stack([refs[s]-refs[0.] for s in scales]);c=np.linalg.solve(D,y)
        for scale in [.25,.5,1.,2.]:
            keep=np.ones(1001,bool);t=np.arange(1001)
            for sp in events[scale]:keep &= abs(t-sp)>10
            linear=refs[0.]+scale*c[0];quadratic=linear+scale**2*c[1];ref=refs[scale]
            allrows.append(dict(A=A,scale=scale,reference_spikes=events[scale].tolist(),retained_points=int(keep.sum()),linear_rmse_mv=float(np.sqrt(np.mean((linear-ref)[keep]**2))),quadratic_rmse_mv=float(np.sqrt(np.mean((quadratic-ref)[keep]**2))),response_rms_mv=float(np.sqrt(np.mean((ref-refs[0.])[keep]**2)))))
        np.savez_compressed(out/f'components_A{A}.npz',baseline=refs[0.],first_order=c[0],second_order=c[1])
    (out/'summary.json').write_text(json.dumps(dict(profile=a.profile,calibration_scales=[.0625,.125],fit='Two amplitude samples estimate first and second coefficients; evaluated at larger conductances',input_source='Independent validation rasters; first 1 s, held fixed across amplitudes',mask='Reference AP +/-10 ms excluded; counts retained',rows=allrows),indent=2)+'\n')

if __name__=='__main__':main()
