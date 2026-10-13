# -*- coding: utf-8 -*-
"""Finite-strength bilinear law checks in the unchanged active mouse L5PC.

This measures the original EPSP/IPSP-product law, using full isolated responses.
It is deliberately separate from the constructed first-order-kernel DBNN control.
No pretrained DBNN weights or active channels are altered.
"""
import argparse,json,time,sys,hashlib,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import numpy as np

STATE=None
FIT=[.25,.5,1.];TEST=[1.5,2.,3.,4.,6.,8.,12.,16.]


def init(root,target_rest=None,dt=.1,fixed_bias=None):
    global STATE
    root=Path(root);sys.path.insert(0,str(root/'scripts'))
    import sample_dbnn_data_repro as sample
    from neuron import h,coreneuron
    cfg=dict(root=str(root/'source/ShaiEtAl2015'),biophys_file='L5PCbiophys5.hoc',morphology_file='cell1.asc',num_synapses=500,
        duration_ms=300.,warmup_ms=500.,dt=.1,sample_interval_ms=.1,input_rate_hz=50.,exc_weight=1.88e-4,inh_weight=8e-5,
        exc_tau1_ms=.5,exc_tau2_ms=5.,inh_tau1_ms=1.,inh_tau2_ms=10.,inh_reversal_mv=-80.,v_init=-75.,spike_threshold=-20.,seed=20260425)
    s=sample.build_worker_network(cfg);coreneuron.enable=False
    pc=h.ParallelContext();pc.nthread(1);pc.thread_busywait(0);pc.set_maxstep(10)
    for ns in s['netstims']:ns.number=0
    for nc in s['netcons']:nc.weight[0]=0
    active=[h.NetCon(None,syn) for syn in s['synapses']]
    for nc in active:nc.delay=0
    h.distance(0,.5,sec=s['cell'].soma[0]);dist=np.array([h.distance(syn.get_segment().x,sec=syn.get_segment().sec) for syn in s['synapses']])
    exc=np.arange(250);inh=np.arange(250,500)
    groups={
      'proximal':(exc[np.argsort(dist[exc])[:32]],inh[np.argsort(dist[inh])[:32]]),
      'distributed':(exc[np.linspace(0,249,32).round().astype(int)],inh[np.linspace(0,249,32).round().astype(int)]),
      'distal':(exc[np.argsort(dist[exc])[-32:]],inh[np.argsort(dist[inh])[-32:]])}
    holding=h.IClamp(s['cell'].soma[0](.5));holding.delay=0;holding.dur=1e9;holding.amp=0
    bias=0.
    if fixed_bias is not None:
        bias=float(fixed_bias);holding.amp=bias
    elif target_rest is not None:
        lo,hi=0.,.15
        h.dt=dt;h.steps_per_ms=1/dt;h.secondorder=0
        def resting_voltage(amp):
            holding.amp=amp;h.finitialize(-75);h.continuerun(500)
            return float(s['cell'].soma[0](.5).v)
        while resting_voltage(hi)<target_rest:
            hi*=2
            assert hi<=.6,'Holding target was not bracketed below 0.6 nA'
        for _ in range(20):
            mid=(lo+hi)/2
            if resting_voltage(mid)<target_rest:lo=mid
            else:hi=mid
        bias=(lo+hi)/2;holding.amp=bias
        assert abs(resting_voltage(bias)-target_rest)<.001
    STATE=dict(holding=holding,holding_current_na=bias,target_rest_mv=target_rest,dt_ms=dt,root=root,h=h,s=s,pc=pc,active=active,groups=groups,dist=dist,cfg=cfg)


def simulate(job):
    s=STATE;h=s['h'];name,se,si=job;ids_e,ids_i=s['groups'][name]
    for nc in s['active']:nc.weight[0]=0
    for i in ids_e:s['active'][int(i)].weight[0]=se*1.88e-4
    for i in ids_i:s['active'][int(i)].weight[0]=si*8e-5
    vv=h.Vector().record(s['s']['cell'].soma[0](.5)._ref_v);tt=h.Vector().record(h._ref_t)
    sn=h.NetCon(s['s']['cell'].soma[0](.5)._ref_v,None,sec=s['s']['cell'].soma[0]);sn.threshold=-20
    sp=h.Vector();sn.record(sp)
    h.dt=s['dt_ms'];h.steps_per_ms=1/s['dt_ms'];h.secondorder=0;h.finitialize(-75);h.fcurrent();h.frecord_init();h.continuerun(500)
    for i in ids_e:
        if se:s['active'][int(i)].event(520.)
    for i in ids_i:
        if si:s['active'][int(i)].event(520.)
    h.continuerun(800)
    t=np.arange(3001)*.1;voltage=np.interp(500+t,np.array(tt.as_numpy(),copy=True),np.array(vv.as_numpy(),copy=True))
    events=np.array(sp.as_numpy(),copy=True);events=events[(events>=500)&(events<=800)]-500
    return dict(profile=name,strength_e=se,strength_i=si,time_ms=t,voltage_mv=voltage,spikes_ms=events,
                holding_current_na=s['holding_current_na'],ids_e=ids_e,ids_i=ids_i,distance_e_um=s['dist'][ids_e],distance_i_um=s['dist'][ids_i])


def key(name,e,i):return f'{name}_E{e:g}_I{i:g}'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--target-rest',type=float);ap.add_argument('--dt',type=float,default=.1);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    tasks=[]
    for name in ['proximal','distributed','distal']:
        jobs={(0.,0.)}
        for e in FIT:
            for i in FIT:jobs.add((e,i))
        for e in FIT+TEST:jobs.add((e,0.));jobs.add((0.,e))
        for eps in TEST:jobs.add((eps,eps))
        tasks.extend((name,e,i) for e,i in sorted(jobs))
    start=time.perf_counter();all_results={}
    with ProcessPoolExecutor(a.workers,mp_context=mp.get_context('spawn'),initializer=init,initargs=(str(a.root),a.target_rest,a.dt)) as pool:
        futures=[pool.submit(simulate,j) for j in tasks]
        for future in as_completed(futures):
            r=future.result();k=key(r['profile'],r['strength_e'],r['strength_i']);all_results[k]=r
            np.savez_compressed(a.out/(k+'.npz'),**{k:v for k,v in r.items() if k!='profile'})
            if len(all_results)%12==0:print(f'{len(all_results)}/{len(tasks)} L5PC trajectories completed',flush=True)
    rows=[];fits=[];raw={}
    for name in ['proximal','distributed','distal']:
        base=all_results[key(name,0.,0.)]['voltage_mv'];t=all_results[key(name,0.,0.)]['time_ms'];w=np.ones(len(t));w[[0,-1]]=.5;w/=w.sum()
        fit_product=[];fit_sc=[];fit_conditions=[]
        for e in FIT:
            for i in FIT:
                rr=all_results[key(name,e,i)];er=all_results[key(name,e,0.)];ir=all_results[key(name,0.,i)]
                if len(rr['spikes_ms']) or len(er['spikes_ms']) or len(ir['spikes_ms']):continue
                ve=er['voltage_mv']-base;vi=ir['voltage_mv']-base;full=rr['voltage_mv']-base
                fit_product.append(ve*vi);fit_sc.append(full-ve-vi);fit_conditions.append([e,i])
        p=np.array(fit_product).T;sc=np.array(fit_sc).T
        energy=float(np.sum(w[:,None]*p*p));assert energy>0
        fixed_c=float(np.sum(w[:,None]*p*sc)/energy)
        fit_rm=float(np.sqrt(np.sum(w[:,None]*(sc-fixed_c*p)**2)/p.shape[1]));product_rms=np.sqrt(energy/p.shape[1])
        scale=np.max(np.abs(p),axis=1);support=scale>max(scale.max()*1e-5,1e-30)
        kt=np.full(len(t),np.nan);pn=p[support]/scale[support,None]
        kt[support]=np.sum(pn*sc[support],axis=1)/(scale[support]*np.sum(pn*pn,axis=1))
        temporal_variance=float(np.sum(w[support,None]*p[support]**2*(kt[support,None]-fixed_c)**2)/energy)
        metadata=all_results[key(name,0.,0.)]
        fits.append(dict(profile=name,fixed_c_per_mv=fixed_c,fit_strength_pairs=fit_conditions,
            fit_sc_rmse_mv=fit_rm,product_rms_mv2=float(product_rms),equivalent_total_dispersion_per_mv=fit_rm/product_rms,
            fitted_temporal_std_per_mv=np.sqrt(temporal_variance),resting_soma_mv=float(base[0]),
            ids_e=metadata['ids_e'].tolist(),ids_i=metadata['ids_i'].tolist(),
            holding_current_na=float(metadata['holding_current_na']),mean_distance_e_um=float(metadata['distance_e_um'].mean()),mean_distance_i_um=float(metadata['distance_i_um'].mean())))
        raw[name+'_time_ms']=t;raw[name+'_fitted_effective_kappa']=kt;raw[name+'_fixed_c']=np.array(fixed_c)
        for eps in TEST:
            rr=all_results[key(name,eps,eps)];er=all_results[key(name,eps,0.)];ir=all_results[key(name,0.,eps)]
            ve=er['voltage_mv']-base;vi=ir['voltage_mv']-base;full=rr['voltage_mv']-base
            additive=ve+vi;bilinear=additive+fixed_c*ve*vi;den=float(np.sqrt(w@full**2))
            linear_rm=float(np.sqrt(w@(additive-full)**2));bilinear_rm=float(np.sqrt(w@(bilinear-full)**2));ep=float(ve.max());ip=float(-vi.min())
            spikes=sum(len(r['spikes_ms']) for r in [rr,er,ir]);rows.append(dict(profile=name,strength=float(eps),epsp_peak_mv=ep,ipsp_abs_peak_mv=ip,
                joint_response_rms_mv=den,additive_rmse_mv=linear_rm,fixed_bilinear_rmse_mv=bilinear_rm,
                additive_relative_rms=linear_rm/max(den,1e-15),fixed_bilinear_relative_rms=bilinear_rm/max(den,1e-15),
                peak_total_input_conductance_bound_ns=float(eps*(32*.188+32*.08)),
                all_reference_events=spikes,subthreshold=(spikes==0),
                physiological_band=(spikes==0 and .2<=ep<=10 and ip<=3),common_band=(spikes==0 and .2<=ep<=6 and ip<=3)))
            raw[name+f'_strength_{eps:g}_isolated_and_joint_mv']=np.array([ve,vi,full,additive,bilinear])
    data=dict(finished=True,execution_host=platform.node(),wall_s=time.perf_counter()-start,trajectories=len(all_results),workers=a.workers,
        model='Shai et al. 2015 mouse V1 L5PC, cell1.asc, L5PCbiophys5.hoc, original active channels',
        holding_voltage_target_mv=a.target_rest,solver_dt_ms=a.dt,observation_ms=.1,warmup_ms=500.,duration_ms=300.,onsets_ms=[20.,20.],
        synapses=dict(mechanism='Exp2Syn',candidate_sites=500,sites_in_each_E_or_I_group=32,seed=20260425,
            baseline_peak_conductance_ns=[.188,.08],reversal_mv=[0.,-80.],rise_ms=[.5,1.],decay_ms=[5.,10.]),
        approximation_scope='Companion test of measured finite-response EPSP/IPSP bilinear integration; additive baseline uses full isolated responses, not Taylor first order. Readout c fixed from nine fit-strength pairs and evaluated on eight held-out amplitudes. No DBNN checkpoint fitting.',
        fit_profiles=fits,rows=rows,input_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.out/'l5pc_finite_strength.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda x:x.item())+'\n')
    np.savez_compressed(a.out/'l5pc_finite_strength_summary.npz',**raw)
    print(json.dumps(dict(finished=True,host=data['execution_host'],wall_s=data['wall_s'],trajectories=len(all_results),rows=rows),indent=2))

if __name__=='__main__':main()
