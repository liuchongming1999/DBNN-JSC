"""One detailed L5PC neuron, identical event raster, explicit solver resources."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

HERE=Path(__file__).resolve().parent
SCRIPT_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))


def stamp():return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def stats(v):
    return dict(median_ms=float(np.median(v)),mean_ms=float(np.mean(v)),
                std_ms=float(np.std(v,ddof=1)) if len(v)>1 else 0.,
                p95_ms=float(np.percentile(v,95)),n=len(v),runs_ms=v)


def inputs(A,T,samples,rate='sampled'):
    ids=np.sort(np.random.default_rng(918+A+500).choice(500,A,replace=False))
    with np.load(ROOT/'data/shai_ei_500syn_6s_dataset.npz',allow_pickle=True) as d:
        x=d['test_ip'][samples,:,:T][:,ids].astype(np.float32)
    if rate=='quiet':x.fill(0)
    elif rate=='low':
        rng=np.random.default_rng(1629)
        x*=rng.random(x.shape)<.1
    elif rate=='high':
        rng=np.random.default_rng(2629)
        x=np.maximum(x,(rng.random(x.shape)<.05).astype(np.float32))
    events=[[np.flatnonzero(row).astype(float)+500 for row in trial] for trial in x]
    return ids,x,events


def run(a,repeat_barrier=None):
    os.environ['CORENEURONLIB']=str(HERE/'model/x86_64/libcorenrnmech.so')
    import neuron
    from neuron import h,coreneuron
    import sample_dbnn_data_repro as sample
    ids,x,events=inputs(a.active,a.steps,a.samples,a.rate)
    pc=h.ParallelContext();pc.thread_busywait(a.busywait);pc.nthread(a.threads)
    model_root=ROOT/'source/ShaiEtAl2015' if a.mechanisms=='original' else HERE/'model'
    cfg=dict(root=str(model_root),biophys_file='L5PCbiophys5.hoc',morphology_file='cell1.asc',
             num_synapses=500,duration_ms=a.steps-1,warmup_ms=500.,dt=a.dt,sample_interval_ms=1.,
             input_rate_hz=50.,exc_weight=1.88e-4,inh_weight=8e-5,exc_tau1_ms=.5,exc_tau2_ms=5.,
             inh_tau1_ms=1.,inh_tau2_ms=10.,inh_reversal_mv=-80.,v_init=-75.,spike_threshold=-20.,seed=20260425)
    t=time.perf_counter();s=sample.build_worker_network(cfg)
    for ns in s['netstims']:ns.number=0
    for nc in s['netcons']:nc.weight[0]=0
    syns=[s['synapses'][int(i)] for i in ids]
    s['netstims']=[];s['netcons']=[];s['synapses']=syns
    del ns,nc
    ncs=[]
    for i,syn in zip(ids,syns):
        # Explicit NetCon.event(t) gives the delivery time and ignores delay.
        # A positive metadata delay permits NEURON's parallel event machinery.
        nc=h.NetCon(None,syn);nc.delay=1;nc.weight[0]=cfg['exc_weight'] if i<250 else cfg['inh_weight'];ncs.append(nc)
    spike_nc=h.NetCon(s['cell'].soma[0](.5)._ref_v,None,sec=s['cell'].soma[0]);spike_nc.threshold=-20
    if a.engine=='core':
        pc.set_gid2node(0,0);pc.cell(0,spike_nc)
    cv=h.CVode();cv.cache_efficient(a.cache)
    if a.engine=='multisplit':
        h.load_file('parcom.hoc')
        par=h.ParallelComputeTool();par.multisplit(1)
    else:par=None
    cv.active(a.engine=='cvode')
    if a.engine=='cvode':cv.atol(a.atol)
    # Explicit thread association is required for global time recording after multisplit.
    record_anchor=h.IClamp(s['cell'].soma[0](.5));record_anchor.amp=0;record_anchor.dur=0
    if a.engine in ('core','multisplit'):
        # CoreNEURON 9.0.2 returns fixed-step record vectors, but not the
        # custom-interval record event used by NEURON. Downsample by time below.
        vv=h.Vector().record(s['cell'].soma[0](.5)._ref_v)
        tt=h.Vector().record(h._ref_t)
    else:
        vv=h.Vector().record(record_anchor,s['cell'].soma[0](.5)._ref_v,1.)
        tt=h.Vector().record(h._ref_t,1.) if a.threads==1 else h.Vector().record(record_anchor,h._ref_t,1.)
    spikes=h.Vector();spike_nc.record(spikes)
    h.dt=a.dt;h.steps_per_ms=1/a.dt;h.secondorder=0
    pc.set_maxstep(10)
    pc.thread_stat()
    saved=None
    if getattr(a,'state_cache',0):
        h.finitialize(-75.);h.fcurrent();h.frecord_init();h.continuerun(500.)
        saved=h.SaveState();saved.save()
    build_ms=(time.perf_counter()-t)*1000
    timing={k:[] for k in ('solver_call','warmup','event_schedule','output_copy','trial_e2e','cpu_time')}
    outputs=[];spike_arrays=[];node_counts={};endpoint_appends=0;repeat_windows=[]
    for rep in range(a.repeats+1):
        if repeat_barrier is not None:repeat_barrier.wait(timeout=300)
        rep_start=time.perf_counter()
        totals={k:0. for k in timing};last=[];last_spikes=[]
        for b in range(len(a.samples)):
            begin=time.perf_counter();c0=time.process_time();coreneuron.enable=False
            t=time.perf_counter();h.dt=a.dt
            if saved is None:
                h.finitialize(-75.);h.fcurrent();h.frecord_init();h.continuerun(500.)
            else:
                saved.restore();h.fcurrent();h.frecord_init();spikes.resize(0)
            totals['warmup']+=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            for nc,ts in zip(ncs,events[b]):
                for event in ts:nc.event(float(event))
            totals['event_schedule']+=(time.perf_counter()-t)*1000
            coreneuron.enable=a.engine=='core'
            coreneuron.gpu=bool(a.gpu)
            t=time.perf_counter();pc.psolve(500.+a.steps-1)
            totals['solver_call']+=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            times=np.array(tt.as_numpy(),dtype=np.float64,copy=True)
            vals=np.array(vv.as_numpy(),dtype=np.float64,copy=True)
            mask=(times>=500-1e-6)&(times<=500+a.steps-1+1e-6)
            if a.engine in ('core','multisplit'):mask &= np.abs(times-np.rint(times))<1e-5
            vals=vals[mask];times=times[mask]
            endpoint=500.+a.steps-1
            if len(vals)==a.steps-1 and abs(float(h.t)-endpoint)<1e-5:
                # A recording event exactly at psolve(tstop) may remain queued.
                # Read that same terminal state, without advancing physical time.
                vals=np.append(vals,float(s['cell'].soma[0](.5).v));times=np.append(times,endpoint)
                endpoint_appends+=1
            assert vals.shape==(a.steps,), (vals.shape,a.steps,times[:3],times[-3:])
            assert np.isfinite(vals).all()
            assert np.max(np.abs(times-(500+np.arange(a.steps))))<1e-5
            last.append(vals)
            sp=np.array(spikes.as_numpy(),dtype=np.float64,copy=True)
            last_spikes.append(sp[sp>=500]-500)
            totals['output_copy']+=(time.perf_counter()-t)*1000
            totals['cpu_time']+=(time.process_time()-c0)*1000
            totals['trial_e2e']+=(time.perf_counter()-begin)*1000
        node_counts=Counter()
        for sec in h.allsec():node_counts[int(pc.sec_in_thread(sec=sec))]+=int(sec.nseg)
        if rep:
            for k in timing:timing[k].append(totals[k])
            outputs.append(np.stack(last));spike_arrays.append([v.tolist() for v in last_spikes])
            repeat_windows.append([rep_start,time.perf_counter()])
        print(f'{a.engine} nt={a.threads} rep={rep} solve_ms={totals["solver_call"]:.3f} nodes={dict(node_counts)}',flush=True)
    result=dict(finished=True,created=stamp(),config=vars(a).copy(),neuron=neuron.__version__,build=h.nrnversion(6),
        cell_count=1,compartments=sum(node_counts.values()),sections=sum(1 for _ in h.allsec()),
        compartments_by_thread=dict(node_counts),input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
        active_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),input_events=int(np.count_nonzero(x)),
        affinity=sorted(os.sched_getaffinity(0)),loadavg=os.getloadavg(),build_ms=build_ms,
        timing={k:stats(v) for k,v in timing.items()},peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        spikes=spike_arrays,voltage_repeat_max_diff_mv=float(np.max(np.abs(np.asarray(outputs)-outputs[0]))),
        worker_sha256=SCRIPT_SHA256,recording_mode='internal_dt_then_1ms' if a.engine in ('core','multisplit') else '1ms_record',
        timer_scope='solver_call includes soma recording; CoreNEURON online transfer included; trial_e2e includes warmup/event scheduling/output copy, model build separate',
        thread_busywait=a.busywait,actual_threads=int(pc.nthread()),core_gpu_build_available='CORENRN_ENABLE_GPU=ON' in h.nrnversion(6))
    result['terminal_state_appends']=endpoint_appends
    result['repeat_wall_windows']=repeat_windows
    result['thread_integration_seconds']=[float(pc.thread_ctime(i)) for i in range(int(pc.nthread()))]
    if par is not None:result['multisplit']=dict(pieces=float(par.lb.npiece_),load_balance=float(par.lb.thread_cxbal_))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out.with_suffix('.npz'),voltage=np.asarray(outputs),ids=ids,spike_times=np.array(spike_arrays,dtype=object))
    result['voltage_file']=out.with_suffix('.npz').name
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--engine',choices=['neuron','multisplit','cvode','core'],default='neuron')
    ap.add_argument('--threads',type=int,default=1);ap.add_argument('--busywait',type=int,default=0)
    ap.add_argument('--cache',type=int,default=1);ap.add_argument('--dt',type=float,default=.1)
    ap.add_argument('--state-cache',type=int,default=0)
    ap.add_argument('--atol',type=float,default=1e-4);ap.add_argument('--gpu',action='store_true')
    ap.add_argument('--mechanisms',choices=['original','core'],default='core')
    ap.add_argument('--active',type=int,default=500);ap.add_argument('--steps',type=int,default=1001)
    ap.add_argument('--samples',type=int,nargs='+',default=[0]);ap.add_argument('--rate',choices=['sampled','quiet','low','high'],default='sampled')
    ap.add_argument('--repeats',type=int,default=3);ap.add_argument('--output',required=True)
    a=ap.parse_args();a.output=str(Path(a.output).resolve())
    try:r=run(a)
    except Exception as e:
        r=dict(finished=False,created=stamp(),config=vars(a),error=repr(e),traceback=traceback.format_exc());print(r['traceback'],flush=True)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n')
    if not r['finished']:sys.exit(1)


if __name__=='__main__':main()
