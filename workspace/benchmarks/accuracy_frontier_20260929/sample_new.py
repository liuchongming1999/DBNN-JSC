"""Independent half-ms event panel for accuracy development; no main31 targets."""
import argparse, hashlib, multiprocessing as mp, sys
from datetime import datetime
from pathlib import Path
import numpy as np
from common import read, write
STATE = {}

def initialize(root, out):
    root=Path(root);out=Path(out)
    sys.path.insert(0,str(root/'benchmarks/training_refinement_20260929'))
    import sample_training_panel as old
    from reference_fresh import run_case
    old.initialize(root,out)
    STATE.update(old.STATE);STATE.update(run_case=run_case,out=out)

def run(row):
    s=STATE
    r=s['run_case'](s['h'],s['state'],s['ncs'],row,s['out'],s['out']/'references')
    return dict(case_id=row['case_id'],spikes=r['spike_count'],wall_ms=r['wall_ms'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--workers',type=int,default=6)
    a=ap.parse_args();root=a.root.resolve();out=a.work.resolve()/'fresh'
    (out/'inputs').mkdir(parents=True,exist_ok=True);(out/'references').mkdir(exist_ok=True)
    sys.path.insert(0,str(root/'benchmarks/neuron_single_expanded_20260929'))
    from generate_manifest import pick_sites
    src=np.load(root/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True)
    rows=[]
    for A in [64,128,256,500]:
        for hz in [10,25,50]:
            for rep in range(8):
                seed=291040000+A*1000+hz*100+rep
                mode=['random','stratified','proximal','distal','random','stratified','random','distal'][rep]
                ids=pick_sites(src['channel_type'],src['channel_location'],A,mode if A<500 else 'random',seed).astype(np.int32)
                rng=np.random.default_rng(seed+593)
                x=rng.poisson(hz*.0005,(A,12001)).astype(np.uint8);x[:,0]=0;x[:,-1]=0
                cid=f'fresh_A{A}_hz{hz}_r{rep}';ip=out/'inputs'/(cid+'.npz')
                np.savez_compressed(ip,x=x,ids=ids)
                rows.append(dict(case_id=cid,input_file=str(ip),split='validation' if rep>=6 else 'fit',
                    A=A,rate_hz=hz,site_mode=mode if A<500 else 'all',duration_ms=6000,
                    input_interval_ms=.5,input_events=int(x.sum()),seed=seed,
                    input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
                    reference_file=str(out/'references'/(cid+'_reference.npz'))))
    forbidden=read(root/'benchmarks/neuron_single_expanded_20260929/selected_20260929/manifest.json')['cases']
    assert not {r['input_sha256'] for r in rows}&{r['input_sha256'] for r in forbidden}
    write(out/'manifest.json',dict(cases=rows,created=datetime.now().isoformat(),fit=72,validation=24,
        neuron_dt_ms=.1,output_interval_ms=.5,input_interval_ms=.5,
        protocol='Fresh Poisson counts at half-ms bins; preserved coincident event multiplicity; fixed full active channels and original synapse positions; no main31 used.'))
    records=[]
    with mp.get_context('spawn').Pool(a.workers,initializer=initialize,initargs=(str(root),str(out))) as pool:
        for rec in pool.imap_unordered(run,rows,chunksize=1):
            records.append(rec);write(out/'status.json',dict(completed=len(records),total=len(rows),latest=rec))
            print('SAMPLED',len(records),len(rows),rec,flush=True)
    write(out/'complete.json',dict(finished=True,records=records));print('COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
