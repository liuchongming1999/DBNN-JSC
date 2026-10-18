"""Error decomposition of the actual frozen primary DBNN at original rest.

Reuse the prespecified 16-site inputs, keep original active channels, no
holding current, original 500-ms warm-up. Readouts computed by LS are offline
diagnostic arrays only: they never replace any parameter in the checkpoint.
The active-cell expansion is an empirical derivative diagnostic, not an
application of the passive stability theorem to an active cell.
"""
from pathlib import Path
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import platform
import sys
import time

import numpy as np
import torch

import passive_diagnostic as core

STATE = None
CP_SHA = '737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d'


def init_active(root, dt):
    global STATE
    root=Path(root); sys.path.insert(0,str(root/'scripts'))
    import sample_dbnn_data_repro as sampler
    from neuron import h, coreneuron
    net=sampler.build_worker_network(core.config(root)); coreneuron.enable=False
    pc=h.ParallelContext(); pc.nthread(1); pc.thread_busywait(0)
    for ns in net['netstims']: ns.number=0
    for nc in net['netcons']: nc.weight[0]=0
    h.distance(0,.5,sec=net['cell'].soma[0])
    dist=np.array([h.distance(s.get_segment().x,sec=s.get_segment().sec) for s in net['synapses']])
    ids=[]
    for candidates in [np.arange(250),np.arange(250,500)]:
        order=candidates[np.argsort(dist[candidates],kind='stable')]
        ids+=order[np.linspace(0,249,8).round().astype(int)].tolist()
    direct=[h.NetCon(None,net['synapses'][i]) for i in ids]
    for nc in direct: nc.delay=0
    STATE=dict(h=h,net=net,pc=pc,direct=direct,dt=dt,ids=ids,
               metadata=dict(site_ids=ids,candidate_sites=500,selected_sites=16,
                             distance_um=dist[ids].tolist(),warmup_ms=500.,v_init_mv=-75.,
                             holding_current_na=0.,active_channels='Original unmodified L5PCbiophys5.hoc',
                             section_names=net['channel_section'][ids].tolist(),
                             section_locations=net['channel_location'][ids].tolist()))


def simulate_active(job):
    name,x,eps=job; s=STATE; h=s['h']
    for j,nc in enumerate(s['direct']): nc.weight[0]=eps*(1.88e-4 if j<8 else 8e-5)
    v=h.Vector().record(s['net']['cell'].soma[0](.5)._ref_v,1.)
    tt=h.Vector().record(h._ref_t,1.)
    h.dt=s['dt']; h.steps_per_ms=1/h.dt; h.secondorder=0; h.cvode_active(0)
    h.finitialize(-75.); h.fcurrent(); h.frecord_init()
    for j,nc in enumerate(s['direct']):
        for onset in np.flatnonzero(x[j]):nc.event(500.+float(onset))
    h.continuerun(1000.)
    voltage=np.interp(500.+np.arange(core.N),np.array(tt.as_numpy(),copy=True),np.array(v.as_numpy(),copy=True))
    assert np.isfinite(voltage).all()
    return name,eps,voltage


def load_active(out,dt,row,eps):
    bg=np.load(core.cache_path(out,dt,'zero',0.))['voltage_mv']
    v=np.load(core.cache_path(out,dt,row['name'],float(eps)))['voltage_mv']
    return v-bg


def make_baseline(root,out,dt):
    p=core.cache_path(out,dt,'zero',0.)
    if p.exists():return
    init_active(root,dt)
    _,_,v=simulate_active(('zero',np.zeros((16,core.N),np.uint8),0.))
    p.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(p,voltage_mv=v,epsilon=0.,dt_ms=dt)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True); ap.add_argument('--workers',type=int,default=8)
    args=ap.parse_args(); root=args.root.resolve(); out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    assert platform.node()=='ps'
    guard=open(out/'run.lock','w'); fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic(); torch.set_num_threads(1)
    cp=root/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt'
    assert core.sha(cp)==CP_SHA
    state=torch.load(cp,map_location='cpu',weights_only=False)['model_state']
    state={k:v.double().numpy() for k,v in state.items()}
    init_active(root,.05); metadata=STATE['metadata']; ids=np.array(metadata['site_ids'])
    kernels=[dict(physical_impulse_amplitude_mv=float(state['omega'][i]*(1 if i<250 else -1)),
                  tau_rise_ms=float(state['tau_rise'][i]),tau_decay_ms=float(state['tau_decay'][i])) for i in ids]
    full=state['bilinear.weight'][0][np.ix_(ids,ids)]
    i,j=np.triu_indices(16); wm=full[i,j]+full[j,i]; wm[i==j]*=.5
    bias=float(state['bilinear.bias'].item())
    rows=core.make_profiles(); mainrows=[r for r in rows if r['split'] in ['fit','test']]
    # Patch only the in-process dispatch API; source files remain unchanged.
    core.init=init_active; core.simulate=simulate_active; core.load=load_active
    protocol=dict(model=metadata,profiles=rows,checkpoint_sha256=CP_SHA,
                  checkpoint=str(cp),all_primary_parameters_frozen=True,
                  solver_dt_ms=.05,control_dt_ms=.025,observation_ms=1.,
                  extraction_scale=.08,extraction_nodes=core.NODES.tolist(),degree=5,
                  epsilon_levels=core.EPS,source_sha256=core.sha(__file__),
                  dependency_sha256=core.sha(Path(core.__file__)),
                  interpretation='Physical coefficients estimated empirically in active NEURON. Counterfactual LS readouts are diagnostic arrays, not trained model replacements. Differences from the fitted W under this diagnostic objective are not optimization errors for the original training objective.')
    if (out/'protocol.json').exists():assert core.read(out/'protocol.json')==protocol
    core.write(out/'protocol.json',protocol)
    make_baseline(root,out,.05)
    core.collect(root,out,rows,core.NODES*.08,.05,args.workers)
    core.collect(root,out,mainrows,core.EPS,.05,args.workers)
    controls=[r for r in rows if r['name'] in ['fit_000','fit_017','fit_033','fit_049','test_000','test_008','test_016','test_017','isolated_000','isolated_015']]
    core.collect(root,out,controls,core.NODES*.04,.05,args.workers)
    make_baseline(root,out,.025)
    core.collect(root,out,controls,list(core.NODES*.08)+[1.,4.],.025,args.workers)
    core.write(out/'status.json',dict(stage='decomposing_frozen_network'))
    data={}; raw={}; isolated=[]
    for r in rows:
        v1,v2,gap=core.coefficients(out,.05,r)
        y=core.features(r,kernels); phi=core.products(y)
        data[r['name']]=dict(v1=v1,v2=v2,y=y,phi=phi)
        if r['split']=='kernel_fit':
            target=v1; pred=y.sum(1)
            isolated.append(dict(name=r['name'],site_id=int(ids[int(r['name'].split('_')[1])]),
                                 rms_mv=core.rms(target-pred),target_rms_mv=core.rms(target),
                                 relative_rms=core.rms(target-pred)/max(core.rms(target),1e-30)))
        raw[r['name']+'_V1']=v1; raw[r['name']+'_V2']=v2;raw[r['name']+'_features']=y
    fitrows=[r for r in mainrows if r['split']=='fit']
    xx=np.concatenate([data[r['name']]['phi'] for r in fitrows])
    target=np.concatenate([data[r['name']]['v2'] for r in fitrows])
    w2,projection,_=core.projection(xx,target)
    xv=[];tv=[]
    for r in fitrows:
        d=data[r['name']]
        for e in [.5,1.,2.]:
            xv.append(e*e*d['phi']);tv.append(load_active(out,.05,r,e)-e*d['y'].sum(1))
    xv=np.concatenate(xv); tv=np.concatenate(tv);wv,vprojection,_=core.projection(xv,tv)
    trained_delta2=core.rms(target-xx@wm)
    orthog=dict(physical_second_projection_mse=core.rms(target-xx@w2)**2,
                original_trained_readout_V2_mse=trained_delta2**2,
                readout_prediction_gap_mse=core.rms(xx@(wm-w2))**2)
    orthog['identity_abs_gap']=abs(orthog['original_trained_readout_V2_mse']-orthog['physical_second_projection_mse']-orthog['readout_prediction_gap_mse'])
    bg=np.load(core.cache_path(out,.05,'zero',0.))['voltage_mv']; a0=bg-bias
    theta=float(state['v_th'].item());amp=float(state['reset_amp'].item());alpha=np.exp(-1/float(state['tau_reset'].item()))
    records=[];checks=[]
    for r in mainrows:
        d=data[r['name']]; first=d['y'].sum(1)
        a=d['v1']-first;b=d['v2']-d['phi']@wm
        for e in core.EPS:
            true=load_active(out,.05,r,e)
            rem=true-e*d['v1']-e*e*d['v2']; actual=e*first+e*e*d['phi']@wm
            gap=float(np.max(abs(true-actual-(e*a+e*e*b+rem))))
            assert gap<1e-10
            drive=bias+actual; closed=np.empty(core.N); reset=0.; events=[]
            for n in range(core.N):
                closed[n]=drive[n]+reset;fire=closed[n]>=theta
                if fire:events.append(float(n))
                reset=alpha*reset-(amp if fire else 0.)
            rec=dict(name=r['name'],family=r['family'],split=r['split'],epsilon=e,
                     delta_background_mv=core.rms(a0),delta1_mv=core.rms(a),
                     delta2_main_mv=core.rms(b),delta2_projected_mv=core.rms(d['v2']-d['phi']@w2),
                     response_rms_mv=core.rms(true),response_peak_abs_mv=float(np.max(abs(true))),
                     first_contribution_mv=e*core.rms(a),second_contribution_mv=e*e*core.rms(b),
                     higher_order_residual_mv=core.rms(rem),
                     main_drive_response_rmse_mv=core.rms(true-actual),
                     main_drive_absolute_rmse_mv=core.rms(true+bg-drive),
                     main_closed_loop_absolute_rmse_mv=core.rms(true+bg-closed),
                     main_predicted_events_ms=events,
                     projected_second_readout_rmse_mv=core.rms(true-e*first-e*e*d['phi']@w2),
                     finite_voltage_conditional_readout_rmse_mv=core.rms(true-e*first-e*e*d['phi']@wv),
                     first_only_rmse_mv=core.rms(true-e*first),
                     physical_second_rmse_mv=core.rms(rem),
                     vector_identity_gap_mv=gap,
                     triangle_bound_mv=e*core.rms(a)+e*e*core.rms(b)+core.rms(rem))
            assert rec['main_drive_response_rmse_mv']<=rec['triangle_bound_mv']+1e-10
            records.append(rec)
        if r in controls:
            for tag,dt,scale,degree in [('half_extraction',.05,.04,5),('degree4',.05,.08,4),('half_step',.025,.08,5)]:
                v1,v2,g=core.coefficients(out,dt,r,scale,degree)
                checks.append(dict(name=r['name'],control=tag,
                                   V1_rms_gap_mv=core.rms(d['v1']-v1),V2_rms_gap_mv=core.rms(d['v2']-v2),
                                   V1_relative_rms_gap=core.rms(d['v1']-v1)/max(core.rms(d['v1']),1e-30),
                                   V2_relative_rms_gap=core.rms(d['v2']-v2)/max(core.rms(d['v2']),1e-30)))
    assert core.sha(cp)==CP_SHA
    np.savez_compressed(out/'arrays.npz',**raw,main_readout=wm,
                        second_projection_readout=w2,finite_voltage_projection_readout=wv,background=bg)
    result=dict(finished=True,wall_s=time.monotonic()-started,execution_host=platform.node(),
                protocol=protocol,primary_checkpoint_unchanged=True,kernels=kernels,
                isolated=isolated,projection=projection,finite_voltage_projection=vprojection,
                fitting_set_V2_orthogonality=orthog,background_bias_mv=bias,
                background_start_mv=float(bg[0]),background_end_mv=float(bg[-1]),
                background_range_mv=float(np.ptp(bg)),records=records,controls=checks,
                all_input_cases_retained=True,
                interpretation=['This is the actual primary checkpoint on declared new inputs without holding current; no retraining or reset calibration.',
                                'Sixteen selected sites are a controlled morphology subset, not a substitute for all main379 inputs.',
                                'The original trained matrix targets finite-strength voltage, not isolated Taylor V2. Its difference from the V2 optimum is a diagnostic objective gap, not proof of failed optimization.',
                                'The finite-voltage conditional LS readout also uses an unweighted diagnostic objective, not the original trajectory-balanced, masked, derivative and regularization objective.',
                                'Active-cell expansion coefficients and remainder are measured numerically; passive theorem constants are not asserted for this model.'])
    core.write(out/'complete.json',result);core.write(out/'status.json',dict(stage='complete',full_results=True))
    print('ACTIVE_ERROR_BUDGET_COMPLETE',len(records),flush=True)


if __name__=='__main__':main()
