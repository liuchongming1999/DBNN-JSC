"""Native-resolution display traces; validate against archived scored outputs."""
from pathlib import Path
import argparse,os,sys,json,hashlib
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--cpu',type=int,required=True);a=p.parse_args()
os.sched_setaffinity(0,{a.cpu})
root=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');work=root/'benchmarks/solver_main379_20261016'
sys.path.insert(0,str(root/'benchmarks/neuron_single_expanded_20260929'));import run_multisplit_grid as base
args=argparse.Namespace(root=root,dt=.0125,threads=1,multisplit=False,verbose=0)
h,pc,state,ncs,snc,spikes,par,meta=base.build(args)
row=next(r for r in json.loads((work/'manifest.json').read_text())['cases'] if r['case_id']==a.case)
with np.load(row['input_file']) as z:ids=z['ids'].copy();raster=z['x'].copy()
assert hashlib.sha256(raster.tobytes()).hexdigest()==row['input_sha256']
out=work/'paper_examples';out.mkdir(exist_ok=True)
reports=[]
for dt in [.0125,.25]:
    for nc in ncs:nc.weight[0]=0.
    for site in ids:ncs[int(site)].weight[0]=1.88e-4 if site<250 else 8e-5
    vv=h.Vector().record(state['cell'].soma[0](.5)._ref_v);tt=h.Vector().record(h._ref_t)
    spikes.resize(0);h.dt=dt;h.steps_per_ms=1/dt;h.secondorder=0
    h.finitialize(-75);h.fcurrent();h.frecord_init();h.continuerun(500)
    for i,site in enumerate(ids):
        for n in np.flatnonzero(raster[i]):ncs[int(site)].event(float(500+n))
    endpoint=500+row['duration_ms'];pc.psolve(endpoint)
    t=np.asarray(tt.as_numpy()).copy();v=np.asarray(vv.as_numpy()).copy()
    if t[-1]<endpoint-1e-5:t=np.append(t,endpoint);v=np.append(v,float(state['cell'].soma[0](.5).v))
    ev=np.asarray(spikes.as_numpy()).copy();ev=ev[(ev>=500-1e-6)&(ev<=endpoint+1e-6)]-500
    common=np.interp(500+np.arange(row['duration_ms']+1),t,v).astype(np.float32)
    with np.load(work/'outputs/reference1'/f'dt{dt:g}'/f'{a.case}.npz') as z:
        gap=float(np.max(np.abs(common-z['voltage'])));assert gap<1e-4
        assert np.array_equal(ev,z['spike_times_ms'])
    take=(t>=500-1e-6)&(t<=endpoint+1e-6)
    np.savez_compressed(out/f'{a.case}_native{dt:g}.npz',time_ms=t[take]-500,voltage=v[take],spike_times_ms=ev)
    reports.append(dict(dt_ms=dt,case_id=a.case,max_common_voltage_gap_mv=gap,event_times_identical=True,native_samples=int(take.sum())))
    print(reports[-1],flush=True)
    del vv,tt
pred=root/'benchmarks/random_init_main379_20261012/predictions'/f'{a.case}.npz'
with np.load(pred) as z:dbnn=z['voltage'].copy()
analysis=json.loads((work/'analysis.json').read_text());gr=next(r for r in analysis['models']['gpu']['records'] if r['case_id']==a.case)
sr=next(r for r in analysis['models']['multisplit8_dt0.25']['records'] if r['case_id']==a.case)
with np.load(work/'outputs/reference1/dt0.0125'/f'{a.case}.npz') as z:rv=z['voltage'].copy()
with np.load(work/'outputs/reference1/dt0.25'/f'{a.case}.npz') as z:sv=z['voltage'].copy()
np.savez_compressed(out/f'{a.case}.npz',time_ms=np.arange(len(dbnn)),dbnn_voltage=dbnn,reference_voltage=rv,neuron_voltage=sv,
    reference_events=gr['reference_events_ms'],dbnn_events=gr['predicted_events_ms'],neuron_events=sr['predicted_events_ms'])
np.savez_compressed(out/f'{a.case}_input.npz',x=raster,ids=ids)
(out/f'{a.case}_validation.json').write_text(json.dumps(reports,indent=2)+'\n')
