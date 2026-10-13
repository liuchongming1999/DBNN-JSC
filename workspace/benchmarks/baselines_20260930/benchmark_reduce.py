"""Default Neuron_Reduce, identical Shai L5PC/events and shared spike mask."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
import numpy as np
from prepare_compat import prepare
from panel import save

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--dt',type=float,choices=[1,.5,.25],required=True);p.add_argument('--repeats',type=int,default=30);p.add_argument('--limit',type=int,default=0);p.add_argument('--publication',action='store_true');a=p.parse_args()
    root=a.root.resolve();work=a.work.resolve();out=work/('publication' if a.publication else 'aligned')/f'reduce_{a.dt:g}ms';out.mkdir(parents=True,exist_ok=True)
    gate=json.loads((work/'original/reduce_legacy/complete.json').read_text());assert gate['passed']
    sys.path[:0]=[str(prepare(work)),str(root/'scripts'),str(root/'benchmarks/resolution_frontier_20260929'),str(root/'benchmarks/training_refinement_20260929')]
    import sample_training_panel as sample
    from neuron import h,coreneuron
    from neuron_reduce import subtree_reductor
    from common import metric,aggregate
    from train_refinement import match_events,event_aggregate
    sample.initialize(root,out);s=sample.STATE;state=s['state'];pc=s['pc'];coreneuron.enable=False
    # Reduction deletes old dendrites. Remap every NetCon that targets them;
    # detach the sampler's unused parallel NetCons before the deletion.
    for c in s['ncs']:c.setpost(None)
    ncs=state['netcons']
    for ns in state['netstims']:ns.number=0
    for nc in state['netcons']:nc.weight[0]=0
    for nc in ncs:nc.weight[0]=0
    ion_constants={}
    for name in ['ena','ek']:
        vals={float(getattr(seg,name)) for sec in h.allsec() for seg in sec if hasattr(seg,name)}
        assert len(vals)==1,('Model has non-uniform reversal potential; requires location-specific transfer',name,vals)
        ion_constants[name]=vals.pop()
    original_segments=sum(sec.nseg for sec in h.allsec());h.finitialize(-75)
    build_start=time.perf_counter()
    reduced,syns,ncs=subtree_reductor(state['cell'],state['synapses'],ncs,0,total_segments_manual=-1)
    # Author wrapper retains the original active soma; the temporary template
    # soma in hoc_model is deliberately deleted at the end of reduction.
    soma=reduced.soma[0];reduced_segments=sum(sec.nseg for sec in h.allsec())
    # Neuron_Reduce excludes ion mechanisms from RANGE copying. The Shai cell
    # explicitly configures ena=50 and ek=-85; keep those same physical values.
    # This is model adaptation, not a reduction hyperparameter or accuracy fit.
    for sec in h.allsec():
        for seg in sec:
            for name,value in ion_constants.items():
                if hasattr(seg,name):setattr(seg,name,value)
    build_ms=(time.perf_counter()-build_start)*1000;assert len(ncs)==500
    # Verify synaptic kinetics were copied exactly for every original site.
    for i,nc in enumerate(ncs):
        syn=nc.syn();expected=(0.,.5,5.) if i<250 else (-80.,1.,10.)
        assert np.allclose([syn.e,syn.tau1,syn.tau2],expected),(i,syn,expected)
    if a.publication:
        new_ncs=[h.NetCon(None,c.syn()) for c in ncs]
        for c in new_ncs:c.delay=10.
        state['netcons'].clear();state['netstims'].clear();ncs=new_ncs
    timing_protocol=dict(method='Neuron_Reduce',reduction_frequency_hz=0.,segments_manual=-1,threads=1,original_segments=original_segments,reduced_segments=reduced_segments,merged_synapses=len(syns),original_input_sites=500,build_ms=build_ms,dt_ms=a.dt,repeats=a.repeats,ion_constants=ion_constants,scope='Prepared solver wall time including voltage recording; excludes model construction, reduction, 500ms warmup, event scheduling and host interpolation.',precision='NEURON double precision',resource='Shared server; CPU affinity 81; 1 thread',library_version=str(h.nrnversion()))
    if a.publication:
        timing_protocol['event_adapter']='Only explicit None-source NetCons; dormant NetStims removed, same as comparison solvers'
    timing_protocol['cpu_affinity']=list(os.sched_getaffinity(0));timing_protocol['resource']='Shared server; declared CPU affinity, one thread'
    save(out/'protocol.json',timing_protocol)
    nc=h.NetCon(soma(.5)._ref_v,None,sec=soma);nc.threshold=-20;sp=h.Vector();nc.record(sp)
    tt=h.Vector().record(h._ref_t);vv=h.Vector().record(soma(.5)._ref_v)
    src=root/'benchmarks/dt_resolution_20260929/test';manifest=json.loads((src/'manifest.json').read_text());rows=manifest['cases']
    if a.limit:rows=rows[:a.limit]
    records=[]
    for row in rows:
        cid=row['case_id'];oj=out/f'{cid}.json';on=out/f'{cid}.npz'
        if oj.exists() and on.exists() and json.loads(oj.read_text()).get('repeats')==a.repeats:
            records.append(json.loads(oj.read_text()));continue
        d=np.load(row['input_file']);ids=d['ids'];raw=d['x'];assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256']
        for c in ncs:c.weight[0]=0
        for site in ids:ncs[int(site)].weight[0]=1.88e-4 if site<250 else 8e-5
        runs=[];end=500.+row['duration_ms']
        for rep in range(a.repeats+1):
            h.dt=a.dt;h.steps_per_ms=1/a.dt;h.secondorder=0;h.finitialize(-75);h.fcurrent();h.frecord_init();sp.resize(0);h.continuerun(500.)
            for j,site in enumerate(ids):
                for k in np.flatnonzero(raw[j]):
                    for _ in range(int(raw[j,k])):ncs[int(site)].event(float(500+k))
            start=time.perf_counter();pc.psolve(end);elapsed=(time.perf_counter()-start)*1000
            if rep:runs.append(elapsed)
        timegrid=500+np.arange(raw.shape[1]);pv=np.interp(timegrid,np.asarray(tt.as_numpy()),np.asarray(vv.as_numpy())).astype(np.float32);ev=np.array(sp.to_python(),dtype=float);ev=ev[(ev>=500-1e-6)&(ev<=end+1e-6)]-500
        ref=np.load(src/'references'/f'{cid}_reference.npz');mm=metric(ref['voltage'][::2],pv,ref['spike_times'],1.);ee=match_events(ref['spike_times'],ev)
        np.savez_compressed(on,voltage=pv,spike_times=ev)
        rec=dict(finished=True,case_id=cid,dt_ms=a.dt,A=len(ids),duration_s=row['duration_ms']/1000,input_sha256=row['input_sha256'],repeats=a.repeats,wall=dict(median_ms=float(np.median(runs)),q1_ms=float(np.quantile(runs,.25)),q3_ms=float(np.quantile(runs,.75)),runs_ms=runs),metrics=mm,events=ee)
        records.append(rec);save(oj,rec);save(out/'status.json',dict(completed=len(records),total=len(rows),last=cid));print('CASE',a.dt,len(records),cid,mm['rmse_mv'],flush=True)
    result=dict(finished=True,full_main31=not a.limit and len(records)==31,records=records,summary=dict(metrics=aggregate([r['metrics'] for r in records]),events=event_aggregate([r['events'] for r in records]),wall_ms=sum(r['wall']['median_ms'] for r in records)),protocol=timing_protocol)
    save(out/('partial.json' if a.limit else 'complete.json'),result)

if __name__=='__main__':main()
