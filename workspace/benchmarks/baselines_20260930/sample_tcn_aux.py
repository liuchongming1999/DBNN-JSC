"""Record dendritic voltage supervision required by the author's default TCN loss.

Use the existing event/split manifest. No new tuning/test samples are created.
NEURON and synapse settings are copied from the locked DBNN campaign. Native
0.1 ms recording is interpolated onto the existing 0.25 ms target grid.
"""
import argparse,hashlib,json,multiprocessing as mp,os,sys,time
from pathlib import Path
import numpy as np
from download_public import save
STATE={}

def init(root,out):
    root=Path(root);out=Path(out);os.sched_setaffinity(0,set(range(82,90)))
    sys.path.insert(0,str(root/'benchmarks/training_refinement_20260929'))
    import sample_training_panel as old
    old.initialize(root,out);s=old.STATE
    cell=s['state']['cell'];segments=[seg for sec in list(cell.basal)+list(cell.apical) for seg in sec]
    STATE.update(s);STATE.update(segments=segments,seg_names=[str(seg) for seg in segments])

def one(row):
    s=STATE;h=s['h'];state=s['state'];out=s['out'];cid=row['case_id'];folder=out/'raw';folder.mkdir(exist_ok=True)
    dest=folder/(cid+'.npy');meta=folder/(cid+'.json')
    if meta.exists() and dest.exists():
        d=json.loads(meta.read_text())
        if d.get('finished'):return d
    d=np.load(row['input_file']);ids=d['ids'];x=d['x']
    assert hashlib.sha256(x.tobytes()).hexdigest()==row['input_sha256']
    for ns in state['netstims']:ns.number=0
    for nc in state['netcons']:nc.weight[0]=0
    for nc in s['ncs']:nc.weight[0]=0
    for site in ids:s['ncs'][int(site)].weight[0]=1.88e-4 if site<250 else 8e-5
    tt=h.Vector().record(h._ref_t);vv=h.Vector().record(state['cell'].soma[0](.5)._ref_v)
    vectors=[h.Vector().record(seg._ref_v) for seg in s['segments']]
    start=time.time();h.dt=.1;h.steps_per_ms=10;h.secondorder=0;h.finitialize(-75);h.fcurrent();h.frecord_init();h.continuerun(500)
    for j,site in enumerate(ids):
        for k in np.flatnonzero(x[j]):
            for _ in range(int(x[j,k])):s['ncs'][int(site)].event(float(500+k*row['input_interval_ms']))
    h.continuerun(500+row['duration_ms']);t=np.asarray(tt);target=500+np.arange(round(row['duration_ms']/.25)+1)*.25
    soma=np.interp(target,t,np.asarray(vv)).astype(np.float32)
    ref=np.load(row['reference_file'])['voltage'];diff=float(np.max(abs(soma-ref)))
    assert diff<1e-4,(cid,diff,'Aux recording changed existing reference')
    matrix=np.lib.format.open_memmap(dest,mode='w+',dtype=np.float32,shape=(len(target),len(vectors)))
    for j,v in enumerate(vectors):matrix[:,j]=np.interp(target,t,np.asarray(v))
    matrix.flush();del matrix
    rec=dict(finished=True,case_id=cid,shape=[len(target),len(vectors)],dt_ms=.25,reference_check_max_abs_mv=diff,elapsed_s=time.time()-start,split=row['split'])
    save(meta,rec);return rec

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();root=a.root.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/'benchmarks/resolution_frontier_20260929/train/manifest.json').read_text());rows=manifest['cases']
    save(out/'manifest.json',manifest);records=[]
    with mp.get_context('spawn').Pool(a.workers,initializer=init,initargs=(str(root),str(out))) as pool:
        for r in pool.imap_unordered(one,rows):
            records.append(r);save(out/'status.json',dict(completed=len(records),total=len(rows),last=r));print('AUX',len(records),len(rows),r['case_id'],flush=True)
    save(out/'complete.json',dict(finished=True,records=records,scope='All 341 existing fit/validation events, 0.25ms dendritic targets; original soma trace checks passed; no main31 or test trajectories.'))

if __name__=='__main__':main()
