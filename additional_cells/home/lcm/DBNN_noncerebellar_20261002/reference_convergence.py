"""Choose a reference step before creating the independent training/test panels."""
import argparse,json,os
from pathlib import Path
import numpy as np
from models import build,run
from screen import write
from common import metric,aggregate

def events(ref,pred):
    used=set();errors=[]
    for t in ref:
        choices=[(abs(float(t)-float(s)),j) for j,s in enumerate(pred) if j not in used]
        if choices:
            gap,j=min(choices)
            if gap<=5:used.add(j);errors.append(gap)
    return dict(tp=len(errors),fp=len(pred)-len(errors),fn=len(ref)-len(errors),max_gap_ms=max(errors,default=0.))

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
    work=Path(__file__).resolve().parent;out=work/'results'/a.model/'screen'
    if (out/'reference_convergence.json').exists():return
    op=json.loads((out/'operating_point.json').read_text());env=build(work,a.model,'ampa')
    rng=np.random.default_rng(61002081);ids=np.arange(500);rows=[]
    for hz,gain,pulses in [(10,1,0),(25,1,1),(50,1,1),(10,2,1),(25,3,2),(50,3,3)]:
        lam=np.full((500,1001),hz/1000.)
        for center in np.linspace(250,750,pulses):lam[:250,round(center):round(center)+40]+=.12
        x=rng.poisson(lam).astype(np.uint8)*gain;x[:,0]=0;x[:,-1]=0
        kwargs=dict(ids=ids,x=x,input_dt=1.,weight=op['weight_us'],holding=op['holding_current_na'],inhibition_ratio=op['inhibition_ratio'])
        rr=run(env,1000.,dt=.025,**kwargs)
        if len(rr['spike_times'])>3 or (len(rr['spike_times'])>1 and np.diff(rr['spike_times']).min()<20):
            rows.append(dict(hz=hz,gain=gain,pulses=pulses,eligible=False,spikes=len(rr['spike_times'])));continue
        t=np.arange(1001);reference=np.interp(t,rr['time_ms'],rr['voltage']);comparisons={}
        for dt in (.05,.1):
            r=run(env,1000.,dt=dt,**kwargs);pred=np.interp(t,r['time_ms'],r['voltage'])
            comparisons[str(dt)]=dict(metrics=metric(reference,pred,rr['spike_times'],1.),events=events(rr['spike_times'],r['spike_times']),solver_ms=r['solver_ms'])
        rows.append(dict(hz=hz,gain=gain,pulses=pulses,eligible=True,spikes=len(rr['spike_times']),fine_solver_ms=rr['solver_ms'],comparisons=comparisons))
        write(out/'reference_convergence_progress.json',dict(records=rows));print(a.model,hz,gain,rows[-1],flush=True)
    eligible=[r for r in rows if r['eligible']];summary={}
    for dt in (.05,.1):
        q=[r['comparisons'][str(dt)] for r in eligible]
        if not q:
            summary[str(dt)]=dict(eligible_count=0,max_case_rmse_mv=None,tp=0,fp=0,fn=0)
            continue
        summary[str(dt)]=dict(metrics=aggregate([v['metrics'] for v in q]),
                             tp=sum(v['events']['tp'] for v in q),fp=sum(v['events']['fp'] for v in q),fn=sum(v['events']['fn'] for v in q),
                             max_case_rmse_mv=max(v['metrics']['rmse_mv'] for v in q),
                             solve_speedup=sum(r['fine_solver_ms'] for r in eligible)/sum(v['solver_ms'] for v in q))
    # Numerical-reference budget is set before surrogate fitting/evaluation.
    # A passive-only probe does not authorize a coarser spiking reference.
    q=summary['0.1'];seen_spiking=any(r['spikes']>0 for r in eligible)
    accepted=seen_spiking and len(eligible)>=3 and q['max_case_rmse_mv'] is not None and q['max_case_rmse_mv']<=.03 and q['fp']==q['fn']==0
    chosen=.1 if accepted else .025
    op['reference_dt_ms']=chosen;op['reference_selection']='Predetermined same-input 0.025/0.05/0.1ms convergence pilot; .1ms requires spiking coverage, masked max-case RMSE <=.03mV and exact +/-5ms event matching.'
    quiet=run(env,2000.,holding=op['holding_current_na'],dt=chosen)
    op['rest_mv']=float(np.mean(np.interp(np.arange(1000,2001),quiet['time_ms'],quiet['voltage'])))
    write(out/'operating_point.json',op)
    write(out/'reference_convergence.json',dict(finished=True,chosen_reference_dt_ms=chosen,
       criterion=op['reference_selection'],spiking_coverage=seen_spiking,records=rows,summary=summary,
       scope='Predetermined pilot before DBNN fitting; existing cerebellar .025ms reference panels retained.'))
    print('REFERENCE_CHOSEN',a.model,chosen,flush=True)

if __name__=='__main__':main()
