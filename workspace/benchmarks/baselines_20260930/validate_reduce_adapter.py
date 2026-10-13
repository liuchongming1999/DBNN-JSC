"""Check the explicit-event NetStim/NetCon adapter before interpreting reduction."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from panel import save

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();root=a.root.resolve();work=a.work.resolve()
    sys.path[:0]=[str(root/'benchmarks/training_refinement_20260929')];import sample_training_panel as sample
    sample.initialize(root,work);s=sample.STATE;h=s['h'];state=s['state'];pc=s['pc'];ncs=state['netcons']
    for ns in state['netstims']:ns.number=0
    for nc in s['ncs']:nc.setpost(None)
    src=root/'benchmarks/dt_resolution_20260929/test';rows=json.loads((src/'manifest.json').read_text())['cases'];chosen=[rows[0],max(rows,key=lambda r:r['input_events']),min(rows,key=lambda r:r['input_events'])]
    vv=h.Vector().record(state['cell'].soma[0](.5)._ref_v);tt=h.Vector().record(h._ref_t);records=[]
    for row in chosen:
        d=np.load(row['input_file']);ids=d['ids'];raw=d['x'];assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256']
        for nc in ncs:nc.weight[0]=0
        for site in ids:ncs[int(site)].weight[0]=1.88e-4 if site<250 else 8e-5
        h.dt=.1;h.steps_per_ms=10;h.secondorder=0;h.finitialize(-75);h.fcurrent();h.frecord_init();h.continuerun(500)
        for j,site in enumerate(ids):
            for k in np.flatnonzero(raw[j]):
                for _ in range(int(raw[j,k])):ncs[int(site)].event(float(500+k))
        pc.psolve(500+row['duration_ms']);pv=np.interp(500+np.arange(raw.shape[1]),np.asarray(tt.as_numpy()),np.asarray(vv.as_numpy()));ref=np.load(src/'references'/f"{row['case_id']}_reference.npz")['voltage'][::2]
        diff=pv-ref;record=dict(case_id=row['case_id'],A=len(ids),duration_ms=row['duration_ms'],events=int(raw.sum()),rmse_mv=float(np.sqrt(np.mean(diff**2))),max_abs_mv=float(np.max(abs(diff))))
        print(record,flush=True);records.append(record)
    passed=all(r['max_abs_mv']<1e-3 for r in records);save(work/'adapter_validation.json',dict(passed=passed,records=records,scope='Unreduced original Shai L5PC at dt0.1ms, explicit events delivered through the original NetStim NetCons; matches the independent main31 reference, including APs. No tuning.'))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
