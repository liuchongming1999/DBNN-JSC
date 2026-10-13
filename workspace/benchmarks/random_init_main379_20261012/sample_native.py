"""Native solver traces for display only, checked against the frozen 1-ms outputs."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import json,sys
from pathlib import Path
import numpy as np

ROOT=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');PANEL=ROOT/'benchmarks/spike_panel_20261009';WORK=ROOT/'benchmarks/random_init_main379_20261012'


def job(task):
    row,dt=task;sys.path.insert(0,str(PANEL));from run_references import initialize
    import run_references
    initialize(str(ROOT),str(PANEL/'main379'));s=run_references.STATE;h=s['h'];inp=np.load(row['input_file']);ids=inp['ids'];raw=inp['x']
    for nc in s['active']:nc.weight[0]=0
    for site in ids:s['active'][int(site)].weight[0]=1.88e-4 if site<250 else 8e-5
    vv=h.Vector().record(s['network']['cell'].soma[0](.5)._ref_v);tt=h.Vector().record(h._ref_t)
    nc=h.NetCon(s['network']['cell'].soma[0](.5)._ref_v,None,sec=s['network']['cell'].soma[0]);nc.threshold=-20;nc.delay=10
    sp=h.Vector();nc.record(sp);h.dt=dt;h.steps_per_ms=1/dt;h.secondorder=0
    h.finitialize(-75);h.fcurrent();h.frecord_init();h.continuerun(500)
    for j,site in enumerate(ids):
        for t in np.flatnonzero(raw[j]):s['active'][int(site)].event(float(500+t))
    s['pc'].psolve(500+row['duration_ms']);t=np.array(tt.as_numpy(),copy=True)-500;v=np.array(vv.as_numpy(),copy=True)
    target=np.arange(row['duration_ms']+1);ev=np.array(sp.as_numpy(),copy=True)-500;ev=ev[(ev>=-1e-6)&(ev<=row['duration_ms']+1e-6)]
    ref=np.load(PANEL/'confirmatory/references'/f'{row["case_id"]}_reference.npz') if dt==.1 else np.load(PANEL/'methods/neuron_dt0.5/raw'/f'neuron_dt0.5_{row["case_id"]}.npz')
    gap=float(np.max(np.abs(np.interp(target,t,v)-ref['voltage'])));assert gap<1e-4
    assert len(ev)==len(ref['spike_times']) and np.allclose(ev,ref['spike_times'],atol=1e-5,rtol=0)
    keep=(t>=-1e-6)&(t<=row['duration_ms']+1e-6)
    folder=WORK/'examples';np.savez_compressed(folder/f'{row["case_id"]}_native{dt:g}.npz',time_ms=t[keep],voltage=v[keep],events_ms=ev)
    return dict(case_id=row['case_id'],dt_ms=dt,common_grid_max_gap_mv=gap,events_equal=True)


if __name__=='__main__':
    ex=json.loads((WORK/'selected_examples.json').read_text());tasks=[(e['dbnn_record'],dt) for e in ex for dt in [.1,.5] if not (WORK/'examples'/f'{e["case_id"]}_native{dt:g}.npz').exists()]
    with ProcessPoolExecutor(4,mp_context=mp.get_context('spawn'),max_tasks_per_child=1) as p:results=list(p.map(job,tasks))
    (WORK/'display_validation.json').write_text(json.dumps(dict(finished=True,records=results,scope='Native curves for visualization only; primary metrics unchanged'),indent=2)+'\n')
    print(results,flush=True)
