"""Fresh quantized-input training/validation panel, disjoint from main31 test inputs."""
import argparse
from datetime import datetime
import hashlib,json,multiprocessing as mp,os,sys,time
from pathlib import Path
import numpy as np
STATE={}

def initialize(root,out):
    root=Path(root);out=Path(out)
    sys.path[:0]=[str(root/'scripts'),str(root/'benchmarks/neuron_single_expanded_20260929')]
    import sample_dbnn_data_repro as sample
    from neuron import h,coreneuron
    from run_reference import run_case
    cfg=dict(root=str(root/'source/ShaiEtAl2015'),biophys_file='L5PCbiophys5.hoc',morphology_file='cell1.asc',num_synapses=500,duration_ms=6000.,warmup_ms=500.,dt=.1,sample_interval_ms=1.,input_rate_hz=50.,exc_weight=1.88e-4,inh_weight=8e-5,exc_tau1_ms=.5,exc_tau2_ms=5.,inh_tau1_ms=1.,inh_tau2_ms=10.,inh_reversal_mv=-80.,v_init=-75.,spike_threshold=-20.,seed=20260425)
    state=sample.build_worker_network(cfg);coreneuron.enable=False
    pc=h.ParallelContext();pc.nthread(1);pc.thread_busywait(0);pc.set_maxstep(10)
    ncs=[]
    for syn in state['synapses']:
        nc=h.NetCon(None,syn);nc.delay=10.;ncs.append(nc)
    STATE.update(h=h,state=state,ncs=ncs,run_case=run_case,out=out,pc=pc)

def run(row):
    s=STATE;r=s['run_case'](s['h'],s['state'],s['ncs'],row,s['out'],s['out']/'references')
    return dict(case_id=row['case_id'],split=row['split'],spikes=r['spike_count'],wall_ms=r['wall_ms'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);(out/'references').mkdir(exist_ok=True);(out/'inputs').mkdir(exist_ok=True)
    sys.path[:0]=[str(root/'benchmarks/neuron_single_expanded_20260929')]
    from generate_manifest import pick_sites,make_raster
    import torch
    src=np.load(root/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True);ck=torch.load(root/'source/dbnn_checkpoint.pt',map_location='cpu',weights_only=False)
    fit=set(ck['fit_indices']);val=set(ck['val_indices']);assert len(fit)==80 and len(val)==20 and not fit&val
    rows=[]
    def add(cid,ids,x,split,kind,hz,mode,source_index=None):
        ids=ids.astype(np.int32);x=x.astype(np.uint8);rel='inputs/'+cid+'.npz';np.savez_compressed(out/rel,ids=ids,x=x)
        rows.append(dict(case_id=cid,input_file=rel,split=split,kind=kind,source_index=source_index,A=len(ids),duration_s=6,duration_ms=6000,rate_hz=hz,site_mode=mode,input_events=int(x.sum()),input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
    for i,x in enumerate(src['train_ip']):add(f'original_train_{i:03}',np.arange(500),x,'fit' if i in fit else 'validation','quantized_original',50.,'all',i)
    for A in [64,128,256,500]:
        for hz in [10,25,50]:
            for rep in range(4):
                seed=93017000+A*100+hz*11+rep*1031;mode=['random','stratified','distal','random'][rep] if A<500 else 'all'
                ids=pick_sites(src['channel_type'],src['channel_location'],A,mode if A<500 else 'random',seed)
                x=make_raster(ids,6000,hz,seed+521)
                add(f'new_A{A}_hz{hz}_r{rep}',ids,x,'validation' if rep==3 else 'fit','fresh_generalization',float(hz),mode)
    add('zero_fit',np.arange(500),np.zeros((500,6001),np.uint8),'fit','zero_input_anchor',0.,'all')
    main=json.loads((root/'benchmarks/neuron_single_expanded_20260929/selected_20260929/manifest.json').read_text())
    mainhash={r['input_sha256'] for r in main['cases'] if r['input_events']>0};assert not mainhash&{r['input_sha256'] for r in rows if r['input_events']>0}
    manifest=dict(created=datetime.now().isoformat(),cases=rows,fit_count=sum(r['split']=='fit' for r in rows),validation_count=sum(r['split']=='validation' for r in rows),test_scope='The existing 31-case panel and original 10 held-out samples are not training or selection data.',quantization='Exact 1 ms binary raster replay to NEURON dt=0.1 ms; original fully active channels; warmup 500 ms.',selection_rule='Before test evaluation, select checkpoint+reset on pooled validation masked MSE + 0.5*(1-micro F1 at +/-5 ms). RMSE and both VE variances exclude reference AP +/-10 ms. Select visualization examples only after locking, using actual event matches and voltage error within each spike class; disclose this rule.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    ordered=sorted(rows,key=lambda r:0 if r['split']=='validation' else 1)
    records=[]
    with mp.get_context('spawn').Pool(a.workers,initializer=initialize,initargs=(str(root),str(out))) as pool:
        for r in pool.imap_unordered(run,ordered,chunksize=1):
            records.append(r);(out/'status.json').write_text(json.dumps(dict(completed=len(records),total=len(rows),updated=datetime.now().isoformat(),records=records),indent=2)+'\n')
            if len(records)%5==0:print('sampled',len(records),'/',len(rows),flush=True)
    (out/'complete.json').write_text(json.dumps(dict(finished=True,records=records),indent=2)+'\n');print('COMPLETE',len(records),flush=True)

if __name__=='__main__':main()
