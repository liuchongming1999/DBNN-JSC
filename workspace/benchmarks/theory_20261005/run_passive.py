# -*- coding: utf-8 -*-
"""Finite-strength, amplitude-calibrated controls. Execute simulations on 5090.

All readout coefficients are fitted once to the true second-order coefficients
and held fixed across the strength sweep. No fitting to full voltage at each
strength. The true first-order kernel is used to isolate quadratic expression.
"""
import argparse,json,time,hashlib,platform
from pathlib import Path
import numpy as np

LAM=.05;EE=70.;EI=-10.;CAP_PF=200.;LEAK_NS=10.
EPS=np.unique(np.r_[np.geomspace(.05,12,57),[.1,.25,.5,1,2,4,8,12]])


def profiles():
    rng=np.random.default_rng(61009)
    def events(rate,start,stop):
        ts=start+np.cumsum(rng.exponential(1000/rate,round((stop-start)*rate/1000*3)+50))
        return np.unique(np.round(ts[ts<stop]))
    return [
        dict(id='legacy_fast_pair',label='Synchronous 2/5-ms pulses',T=300.,te=[20.],ti=[20.],tau_e=2.,tau_i=5.,ge=.02,gi=.02),
        dict(id='ampa_pair',label='Synchronous 5/10-ms pulses',T=300.,te=[20.],ti=[20.],tau_e=5.,tau_i=10.,ge=.02,gi=.02),
        dict(id='delayed_pair',label='Inhibition delayed by 15 ms',T=300.,te=[20.],ti=[35.],tau_e=5.,tau_i=10.,ge=.02,gi=.02),
        dict(id='packet_pair',label='20-event packets over 38 ms',T=400.,te=list(np.arange(40.,80.,2.)),ti=list(np.arange(45.,85.,2.)),tau_e=5.,tau_i=10.,ge=.004,gi=.004),
        dict(id='tonic_pair',label='Distributed 1-ms event train',T=400.,te=list(np.arange(20.,320.,1.)),ti=list(np.arange(20.,320.,1.)),tau_e=5.,tau_i=10.,ge=.0005,gi=.0005),
        dict(id='poisson_pair',label='Poisson groups: 100/80 Hz',T=1000.,te=events(100,20,900).tolist(),ti=events(80,20,900).tolist(),tau_e=5.,tau_i=10.,ge=.004,gi=.004),
    ]


def integrate(p,h,obs=.1):
    count=round(p['T']/h);stride=round(obs/h);assert count%stride==0
    impulses=np.zeros((count+1,2))
    for channel,name in enumerate(['te','ti']):
        for e in p[name]:impulses[round(e/h),channel]+=p['ge' if channel==0 else 'gi']
    decay=np.exp(-h/np.array([p['tau_e'],p['tau_i']]));half=np.sqrt(decay)
    g=impulses[0].copy();v=np.zeros((3,len(EPS)));c=np.zeros(5)
    tv=np.arange(count//stride+1)*obs
    trace=np.zeros((len(tv),3,len(EPS)));coef=np.zeros((len(tv),5));conductance=np.zeros((len(tv),2));conductance[0]=g
    def rhs(v,c,g):
        ge,gi=g;ye,yi,qe,qi,qei=c
        fv=np.array([EPS*EE*ge-(LAM+EPS*ge)*v[0],EPS*EI*gi-(LAM+EPS*gi)*v[1],
                     EPS*(EE*ge+EI*gi)-(LAM+EPS*(ge+gi))*v[2]])
        fc=np.array([EE*ge-LAM*ye,EI*gi-LAM*yi,-ge*ye-LAM*qe,-gi*yi-LAM*qi,-ge*yi-gi*ye-LAM*qei])
        return fv,fc
    for n in range(count):
        a,b=rhs(v,c,g);aa,bb=rhs(v+h*a/2,c+h*b/2,g*half)
        aaa,bbb=rhs(v+h*aa/2,c+h*bb/2,g*half);aaaa,bbbb=rhs(v+h*aaa,c+h*bbb,g*decay)
        v+=h*(a+2*aa+2*aaa+aaaa)/6;c+=h*(b+2*bb+2*bbb+bbbb)/6
        g=g*decay+impulses[n+1]
        if (n+1)%stride==0:
            k=(n+1)//stride;trace[k]=v;coef[k]=c;conductance[k]=g
    return tv,trace,coef,conductance


def weights(t):
    w=np.ones(len(t));w[[0,-1]]=.5;return w/w.sum()


def fit_quad(coef,w):
    ye,yi,qe,qi,qei=coef.T;features=np.array([ye**2,ye*yi,yi**2]).T
    target=qe+qi+qei;root=np.sqrt(w)
    c,_,rank,sv=np.linalg.lstsq(features*root[:,None],target*root,rcond=1e-12)
    residual=target-features@c
    return c,features,target,float(np.sqrt(w@residual**2)),int(rank)


def pair_variance(coef,w):
    p=coef[:,0]*coef[:,1];target=coef[:,4]
    en=w@p**2;c=(w@(p*target))/en;d2=float(np.sqrt(w@(target-c*p)**2));rms_p=float(np.sqrt(en))
    return dict(best_cross_coefficient_per_mv=float(c),product_rms_mv2=rms_p,
                weighted_kappa_std_per_mv=d2/rms_p,weighted_kappa_variance_per_mv2=(d2/rms_p)**2,
                pair_second_residual_rmse_mv=d2,pair_second_relative_residual=d2/float(np.sqrt(w@target**2)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();rows=[];raw={};checks=[];configs=profiles()
    for p in configs:
        t,v,co,g=integrate(p,.01);tc,vc,cc,gc=integrate(p,.02);w=weights(t)
        half_gap=float(np.max(np.abs(v-vc)));co_gap=float(np.max(np.abs(co-cc)))
        assert half_gap<1e-6 and co_gap<1e-6,(p['id'],half_gap,co_gap)
        c,feature,v2,d2,rank=fit_quad(co,w);joint1=co[:,0]+co[:,1]
        q=feature@c;info=pair_variance(co,w);Epeak=v[:,0].max(axis=0);Ipeak=-v[:,1].min(axis=0)
        for j,eps in enumerate(EPS):
            truth=v[:,2,j];den=float(np.sqrt(w@truth**2));pred1=eps*joint1;pred2=pred1+eps**2*v2;predq=pred1+eps**2*q
            errors=[float(np.sqrt(w@(z-truth)**2)) for z in [pred1,pred2,predq]]
            maxerrs=[float(np.max(np.abs(z-truth)))/max(float(np.max(abs(truth))),1e-15) for z in [pred1,pred2,predq]]
            lead=eps**2*d2;higher=float(np.sqrt(w@(truth-pred2)**2));res=eps**2*(v2-q);r3=truth-pred2
            cross=float(2*(w@(res*r3)));qerr=errors[2]
            assert abs(qerr**2-(lead**2+higher**2+cross))<1e-10
            rows.append(dict(profile=p['id'],epsilon=float(eps),epsp_peak_mv=float(Epeak[j]),ipsp_abs_peak_mv=float(Ipeak[j]),
                joint_peak_mv=float(np.max(abs(truth))),response_rms_mv=den,
                total_peak_synaptic_conductance_ns=float(eps*g.sum(axis=1).max()*CAP_PF),
                first_rmse_mv=errors[0],exact_second_rmse_mv=errors[1],fixed_dbnn_rmse_mv=errors[2],
                first_relative_rms=errors[0]/den,exact_second_relative_rms=errors[1]/den,fixed_dbnn_relative_rms=errors[2]/den,
                first_relative_max=maxerrs[0],exact_second_relative_max=maxerrs[1],fixed_dbnn_relative_max=maxerrs[2],
                second_representation_mv=float(lead),higher_order_mv=higher,second_to_higher_ratio=float(lead/max(higher,1e-30)),
                signed_residual_cross_energy_mv2=cross,
                amplitude_band=(.2<=Epeak[j]<=10 and Ipeak[j]<=3 and np.max(abs(truth))<=15),
                common_band=(.2<=Epeak[j]<=6 and Ipeak[j]<=3 and np.max(abs(truth))<=15),
                second_needed=errors[0]/den>.05,quadratic_meets_5pct=errors[2]/den<=.05))
        # Effective finite-strength kappa fitted across nearby independent strengths,
        # rather than dividing SC by an individual tiny product. Definitions
        # differ from the theoretical coefficient; the distinction is saved.
        effective=[]
        for center in [.25,1.,4.,8.]:
            ids=np.where((EPS>=center*.7)&(EPS<=center*1.4))[0]
            ps=v[:,0,ids]*v[:,1,ids];sc=v[:,2,ids]-v[:,0,ids]-v[:,1,ids]
            scale=np.max(np.abs(ps),axis=1);valid=scale>=max(scale.max()*1e-5,1e-30)
            kt=np.full(len(t),np.nan);norm=ps[valid]/scale[valid,None]
            kt[valid]=np.sum(norm*sc[valid],axis=1)/(scale[valid]*np.sum(norm**2,axis=1))
            energy=np.sum(w[:,None]*ps**2);cf=float(np.sum(w[:,None]*ps*sc)/energy)
            rm=float(np.sqrt(np.sum(w[:,None]*(sc-cf*ps)**2)/len(ids)))
            effective.append(dict(center_strength=center,strengths=EPS[ids].tolist(),fixed_c_per_mv=cf,pair_effective_residual_rms_mv=rm,
                                  effective_std_per_mv=float(np.sqrt(np.sum(w[valid,None]*ps[valid]**2*(kt[valid,None]-cf)**2)/energy))))
            raw[p['id']+f'_effective_kappa_{center:g}']=kt
        rec=dict(config=p,readout_coefficients=c.tolist(),readout_rank=rank,second_representation_rmse_mv=d2,
                 pair_variance=info,effective_finite_strength_fits=effective,half_step_max_voltage_gap_mv=half_gap,
                 half_step_max_coefficient_gap_mv=co_gap)
        checks.append(rec)
        raw.update({p['id']+'_time_ms':t,p['id']+'_reference_mv':v,p['id']+'_first_second_coefficients_mv':co,
                    p['id']+'_conductance_over_C_per_ms':g,p['id']+'_quad_coefficients_per_mv':c})
        print(p['id'],len(EPS),'strengths, max step gap',half_gap,flush=True)
    # Exact finite-strength equilibrium: both relative errors are closed form.
    steady=[]
    for x in [.01,.025,.05,.075,.1,.15,.2,.22360679775,.3,.5]:
        peak=EE*x/(1+x)
        steady.append(dict(conductance_to_leak_ratio=x,conductance_ns=x*LEAK_NS,epsp_mv=peak,
                           first_relative_error=x,full_quadratic_relative_error=x*x))
    counts={}
    for band in ['common_band','amplitude_band']:
        eligible=[r for r in rows if r[band]];need=[r for r in eligible if r['second_needed']];good=[r for r in need if r['quadratic_meets_5pct']]
        counts[band]=dict(grid_points=len(eligible),first_within_5pct=len(eligible)-len(need),second_needed=len(need),
            fixed_quadratic_success=len(good),fixed_quadratic_success_third_remainder_dominant=sum(r['second_to_higher_ratio']<1 for r in good))
    # One common full quadratic readout covers all five 5/10-ms input families.
    shared_profiles=[p['id'] for p in configs if p['id']!='legacy_fast_pair']
    xx=[];zz=[]
    for key in shared_profiles:
        tt=raw[key+'_time_ms'];cc=raw[key+'_first_second_coefficients_mv'];ww=weights(tt)
        ff=np.array([cc[:,0]**2,cc[:,0]*cc[:,1],cc[:,1]**2]).T
        xx.append(ff*np.sqrt(ww[:,None]/len(shared_profiles)));zz.append(cc[:,2:].sum(axis=1)*np.sqrt(ww/len(shared_profiles)))
    common_c,_,common_rank,_=np.linalg.lstsq(np.vstack(xx),np.concatenate(zz),rcond=1e-12)
    for r in rows:
        if r['profile'] not in shared_profiles:continue
        key=r['profile'];tt=raw[key+'_time_ms'];cc=raw[key+'_first_second_coefficients_mv'];ww=weights(tt)
        ff=np.array([cc[:,0]**2,cc[:,0]*cc[:,1],cc[:,1]**2]).T
        j=int(np.argmin(abs(EPS-r['epsilon'])));truth=raw[key+'_reference_mv'][:,2,j]
        pred=r['epsilon']*(cc[:,0]+cc[:,1])+r['epsilon']**2*(ff@common_c)
        r['shared_readout_rmse_mv']=float(np.sqrt(ww@(truth-pred)**2))
        r['shared_readout_relative_rms']=r['shared_readout_rmse_mv']/r['response_rms_mv']
    shared=dict(coefficients_per_mv=common_c.tolist(),rank=int(common_rank),input_profiles=shared_profiles,
                fitting_target='True second-order response across five families with equal profile weights; fixed across all strengths')
    data=dict(finished=True,execution_host=platform.node(),created_epoch=time.time(),wall_s=time.perf_counter()-start,
        scope='Passive isopotential compartment; reset-free; true first-order kernels used as an oracle to isolate full quadratic readout error.',
        parameters=dict(capacitance_pf=CAP_PF,leak_ns=LEAK_NS,tau_membrane_ms=1/LAM,reversal_offsets_mv=[EE,EI],v_rest_mv=-70.),
        metric='Full-window RMS error / RMS of the true above-rest response, NOT absolute membrane voltage and NOT 1-VE. Supplement: maximum absolute error / peak absolute response.',
        strength_scope='All first-order kernels and one full quadratic readout per profile are fixed across the entire strength grid; fitted to true second-order coefficients only.',
        physiological_band_source='Li et al. 2014 Discussion: EPSP <6mV, IPSP magnitude <3mV; near-threshold EPSP ~10mV as an extended comparison.',
        nonlinear_strength_conclusion='With fixed waveforms theoretical weighted kappa variance is amplitude-independent; finite effective kappa can vary with strength.',
        solver=dict(reference_rk4_step_ms=.01,control_rk4_step_ms=.02,observation_ms=.1,discontinuous_input_jumps_handled_at_interval_start=True),
        shared_readout=shared,epsilon_grid=EPS.tolist(),profile_checks=checks,steady_exact=steady,grid_counts_not_population_probabilities=counts,rows=rows,
        input_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'passive_finite_strength.json').write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda x:x.item())+'\n')
    np.savez_compressed(out/'passive_finite_strength.npz',epsilon=EPS,**raw)
    print(json.dumps(dict(finished=True,host=data['execution_host'],wall_s=data['wall_s'],counts=counts),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
