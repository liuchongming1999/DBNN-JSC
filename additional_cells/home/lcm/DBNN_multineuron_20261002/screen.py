"""Reference-only operating-point and held-out quadratic adequacy screening."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
from models import build,run,REGISTRY

def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def window(r,duration,step=.25):return np.interp(np.arange(round(duration/step)+1)*step,r['time_ms'],r['voltage'])
def metric(y,p):
    e=np.asarray(p)-y;return dict(rmse_mv=float(np.sqrt(np.mean(e*e))),ve=float(1-np.var(e)/max(np.var(y),1e-12)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);ap.add_argument('--model',choices=REGISTRY,required=True);ap.add_argument('--profile',choices=['ampa','article_expsyn','ampa_nmda'],default='ampa');a=ap.parse_args()
    work=a.work.resolve();out=work/'results'/a.model/'screen';out.mkdir(parents=True,exist_ok=True)
    if (out/(a.profile+'.json')).exists():return
    env=build(work,a.model,a.profile)
    h=env['h'];dend=env['dend'];positions=env['sites']
    coverage=dict(site_count=500,exc=250,inh=250,dendritic_sections=len(dend),total_dendritic_length_um=float(sum(s.L for s in dend)),
        original_dendritic_segments=sum(s.nseg for s in dend),unique_electrical_segments=len({(p['section_index'],int(p['x']*dend[p['section_index']].nseg)) for p in positions}),
        placement='Length-stratified 250 E and 250 I spatial positions, covering the original dendritic tree; original nseg retained.',sites=positions)
    write(out/'sites.json',coverage)
    if not (out/'operating_point.json').exists():
        native=run(env,2000.);nv=window(native,2000);np.savez_compressed(out/'native_no_input.npz',voltage=nv,spike_times=native['spike_times'])
        holding_trials=[];holding=0.
        for current in [0.,-.005,-.01,-.02,-.04,-.08,-.16,-.32,-.64]:
            r=run(env,2000.,holding=current);v=window(r,2000);last=v[-2001:]
            rec=dict(current_na=current,spikes_last_1000ms=int(np.sum(r['spike_times']>=1000)),rest_mv=float(np.mean(last)),std_mv=float(np.std(last)));holding_trials.append(rec)
            if rec['spikes_last_1000ms']==0 and rec['std_mv']<.03 and -85<rec['rest_mv']<-45:holding=current;break
        else:raise RuntimeError(('No stable quiet operating point',a.model,holding_trials))
        rest=rec['rest_mv'];ids=np.arange(250);rng=np.random.default_rng(61002011);x=rng.poisson(.01,(250,1001)).astype(np.uint8);x[:,0]=0
        r=run(env,1000.,ids,x,weight=1e-6,holding=holding);delta=float(np.mean(window(r,1000)[2000:])-rest)
        weight=float(np.clip(1e-6*2.5/max(delta,.02),1e-7,1e-3))
        op=dict(model=a.model,spec=REGISTRY[a.model],native_spikes_2s=len(native['spike_times']),native_firing_hz=len(native['spike_times'])/2.,
            holding_current_na=holding,holding_trials=holding_trials,rest_mv=rest,weight_us=weight,inhibition_ratio=.5,
            condition='Original active channels and morphology retained. Fixed somatic holding current selects a stable low-spike operating point; identical current and reference state used by all solver comparisons.',
            weight_selection='Reference-only 250E 10Hz mean depolarization approximately 2.5mV, before fitting/testing.',reference_dt_ms=.025,reference_output_interval_ms=.25,warmup_ms=2000.)
        write(out/'operating_point.json',op)
    else:op=json.loads((out/'operating_point.json').read_text())
    holding=op['holding_current_na'];rest=op['rest_mv'];weight=op['weight_us']
    r0=run(env,120.,holding=holding);baseline=window(r0,120)
    # Matched near/far pairs. 8 pairs and 4x4 grid form a screening pilot;
    # these are explicitly distinct from the paper's 50-pair 8x8 analysis.
    pairs=[(i,j) for i,j in [(5,8),(30,33),(80,85),(170,173),(5,200),(20,220),(45,170),(90,245)]]
    rows=[]
    for pidx,(i,j) in enumerate(pairs):
        unit=np.zeros((2,121),np.uint8);unit[:,20]=1
        maxw=weight*.5
        for mult in [1,2,4,8,16,32,64,128]:
            r=run(env,120.,[i,j],unit,weight=weight*mult,holding=holding)
            if len(r['spike_times']) or np.max(window(r,120)-baseline)>12:break
            maxw=weight*mult
        singles={}
        levels=np.array([.2,.4,.7,1.])*maxw
        for site in [i,j]:
            for k,w in enumerate(levels):
                xx=np.zeros((1,121),np.uint8);xx[:,20]=1
                rr=run(env,120.,[site],xx,weight=float(w),holding=holding);singles[site,k]=window(rr,120)-baseline
        target=[];features=[];held=[];linear=[];spikes=0
        for k,w1 in enumerate(levels):
            for l,w2 in enumerate(levels):
                rr=run(env,120.,[i,j],unit,holding=holding,site_weights=[w1,w2]);v=window(rr,120)-baseline
                v1=singles[i,k];v2=singles[j,l];lin=v1+v2
                target.append(v);linear.append(lin);features.append(np.stack([v1*v2,v1*v1*v2,v1*v2*v2],1));held.append(k==l)
                spikes+=len(rr['spike_times'])
        target=np.asarray(target);linear=np.asarray(linear);features=np.asarray(features);held=np.asarray(held)
        valid=np.isfinite(target).all()
        if not valid:raise RuntimeError(('nonfinite pair',pidx))
        tr=~held;models={};coef={}
        for label,n in [('quadratic',1),('cubic',3)]:
            ff=features[tr,:,:n].reshape(-1,n);yy=(target-linear)[tr].reshape(-1);c=np.linalg.lstsq(ff,yy,rcond=None)[0]
            pred=linear+np.einsum('ktd,d->kt',features[:,:,:n],c);coef[label]=c.tolist();models[label]=dict(held_out=metric(target[held],pred[held]),all_grid=metric(target,pred))
        models['linear']=dict(held_out=metric(target[held],linear[held]),all_grid=metric(target,linear))
        np.savez_compressed(out/f'{a.profile}_pair{pidx}.npz',target=target,linear=linear,features=features,held=held,levels=levels,
                            sites=np.array([i,j]),baseline=baseline,time_ms=np.arange(481)*.25)
        rr=dict(pair=[i,j],max_weight_us=maxw,near=pidx<4,suprathreshold_grid_trials=spikes,metrics=models,coefficients=coef);rows.append(rr)
        write(out/(a.profile+'_progress.json'),dict(completed=len(rows),total=len(pairs),latest=rr));print('PAIR',a.model,a.profile,len(rows),models['quadratic']['held_out'],flush=True)
    summary={label:dict(mean_heldout_ve=float(np.mean([r['metrics'][label]['held_out']['ve'] for r in rows])),mean_heldout_rmse_mv=float(np.mean([r['metrics'][label]['held_out']['rmse_mv'] for r in rows]))) for label in ['linear','quadratic','cubic']}
    write(out/(a.profile+'.json'),dict(finished=True,profile=a.profile,operating_point=op,coverage={k:v for k,v in coverage.items() if k!='sites'},pairs=rows,summary=summary,
        protocol='8 predetermined near/far EE pairs, 4x4 strength grid; diagonal strength combinations held out; coefficients fitted on other 12 combinations, applied to full trajectory; low-spike holding operating point. Pilot, not a reproduction of the article figure.',
        nmda_condition='Exp2 AMPA plus deterministic double-exponential NMDA, tau1=2ms tau2=80ms Mg=1, NMDA/AMPA peak conductance=.5' if a.profile=='ampa_nmda' else None))
    print('SCREEN_COMPLETE',a.model,a.profile,summary,flush=True)

if __name__=='__main__':main()
