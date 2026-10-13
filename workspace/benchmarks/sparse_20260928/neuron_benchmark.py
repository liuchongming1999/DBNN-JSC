"""NEURON wall time for the SAME quantized events used by the surrogate.

Core excludes cell construction, warm-up and scheduling. Output recording is
1 ms for both methods. B>1 is a serial single-cell reuse baseline, not a
parallel NEURON throughput claim. Results store timing scopes separately.
"""
import argparse,json,time,sys,gc,subprocess,os
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import sample_dbnn_data_repro as sample
from experiment import CASES,case_data,stats,stamp,measure_group,accuracy
from backends import Prepared,subset_state
from DBNN import DBNN_R,DBNN_F

def run_case(ci,reps):
    torch.set_num_threads(1)
    N,A,T,B=CASES[ci]
    sc,x,ids,mp=case_data(N,A,T,B)
    # Same CPU, same single-thread allocation as NEURON for an algorithm-only comparison.
    cpu_fns={}
    for name,cls in [('pytorch_R_cpu_1t',DBNN_R),('pytorch_F_cpu_1t',DBNN_F)]:
        m=cls(N,T,torch.device('cpu'));m.load_state_dict(sc);m.eval()
        cpu_fns[name]=lambda m=m:m(x)
        if B>1:cpu_fns[name+'_trace_loop']=lambda m=m:torch.cat([m(x[b:b+1]) for b in range(B)])
    p=Prepared(subset_state(sc,ids),x[:,ids].contiguous(),'cpu','cublas')
    cpu_fns['packed_native_cpu_1t']=p
    with torch.no_grad():
        ref=cpu_fns['pytorch_R_cpu_1t']()
        err={name:accuracy(fn(),ref,float(sc['v_th'])) for name,fn in cpu_fns.items()}
        for fn in cpu_fns.values():fn()
        cpu_times=measure_group(cpu_fns,7,gpu=False)
    for name,v in cpu_times.items():v.update(err[name])
    cfg={'root':str(ROOT/'source/ShaiEtAl2015'),'biophys_file':'L5PCbiophys5.hoc','morphology_file':'cell1.asc',
         'num_synapses':N,'duration_ms':T-1.,'warmup_ms':500.,'dt':.1,'sample_interval_ms':1.,'input_rate_hz':50.,
         'exc_weight':1.88e-4,'inh_weight':8e-5,'exc_tau1_ms':.5,'exc_tau2_ms':5.,'inh_tau1_ms':1.,
         'inh_tau2_ms':10.,'inh_reversal_mv':-80.,'v_init':-75.,'spike_threshold':-20.,'seed':20260425}
    start=time.perf_counter();s=sample.build_worker_network(cfg);build_ms=(time.perf_counter()-start)*1000
    h=s['h'];h.cvode_active(0)
    for ns in s['netstims']:ns.number=0
    for nc in s['netcons']:nc.weight[0]=0
    # Candidate positions are a registry, not permanently instantiated synapses.
    # Give NEURON the same active-site optimization (stronger fair baseline).
    active_syn=[s['synapses'][k] for k in ids.tolist()]
    s['netstims']=[];s['netcons']=[];s['synapses']=active_syn
    del ns,nc
    ncs=[]
    for k,syn in zip(ids.tolist(),active_syn):
        nc=h.NetCon(None,syn);nc.delay=0
        nc.weight[0]=cfg['exc_weight'] if k<N//2 else cfg['inh_weight'];ncs.append(nc)
    events=[[torch.nonzero(x[b,k]!=0).flatten().numpy().astype(float)+500 for k in ids] for b in range(B)]
    vv=h.Vector().record(s['cell'].soma[0](.5)._ref_v,1.)
    core=[];warm=[];schedule=[];copy=[];hostscope=[]
    last=[]
    for rep in range(reps+1):
        total_core=total_warm=total_schedule=total_copy=0.;last=[]
        all_start=time.perf_counter()
        for b in range(B):
            h.dt=.1;h.steps_per_ms=10
            t0=time.perf_counter();h.finitialize(-75.);h.fcurrent();h.frecord_init();h.continuerun(500.)
            total_warm+=(time.perf_counter()-t0)*1000
            t0=time.perf_counter()
            for nc,ts in zip(ncs,events[b]):
                for event in ts:nc.event(float(event))
            total_schedule+=(time.perf_counter()-t0)*1000
            t0=time.perf_counter();h.continuerun(500.+T-1)
            total_core+=(time.perf_counter()-t0)*1000
            t0=time.perf_counter();last.append(np.array(vv.as_numpy()[-T:],dtype=np.float32,copy=True))
            total_copy+=(time.perf_counter()-t0)*1000
        total_host=(time.perf_counter()-all_start)*1000
        if rep:
            core.append(total_core);warm.append(total_warm);schedule.append(total_schedule);copy.append(total_copy);hostscope.append(total_host)
        print(f'case={ci} rep={rep} core_ms={total_core:.2f}',flush=True)
    np.savez_compressed(HERE/f'neuron_case{ci}_voltage.npz',voltage=np.stack(last),ids=ids.numpy())
    return {'index':ci,'N':N,'A':A,'T':T,'B':B,'dt_ms':.1,'output_dt_ms':1.,'neuron_threads':1,
       'affinity':sorted(os.sched_getaffinity(0)),'build_ms':build_ms,'core':stats(core),'warmup':stats(warm),
       'event_scheduling':stats(schedule),'output_copy':stats(copy),'warmup_schedule_core_copy':stats(hostscope),
       'events':'same 1-ms quantized raster as DBNN','input_event_count':int((x!=0).sum()),
       'batch_mode':'sequential B trials; no NEURON parallelization','inactive_sites_instantiated':False,
       'voltage_file':f'neuron_case{ci}_voltage.npz','cpu_one_thread':cpu_times,
       'neuron_vs_dbnn_rmse_mv':float(np.sqrt(np.mean((np.stack(last)-ref.numpy())**2))) if N==500 and A==500 else None}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',type=int);ap.add_argument('--repeats',type=int,default=5)
    args=ap.parse_args()
    if args.case is not None:
        r=run_case(args.case,args.repeats);(HERE/f'neuron_case{args.case}.json').write_text(json.dumps(r,indent=2));return
    all_results=[]
    # Cases span full/sparse, N/T/B; separate process avoids residual cells.
    for ci in range(len(CASES)):
        p=HERE/f'neuron_case{ci}.json'
        if not p.exists():
            reps=3 if CASES[ci][3]>1 else args.repeats
            subprocess.run([sys.executable,__file__,'--case',str(ci),'--repeats',str(reps)],check=True)
        all_results.append(json.loads(p.read_text()))
        (HERE/'neuron_results.json').write_text(json.dumps({'created':stamp(),'cases':all_results},indent=2))

if __name__=='__main__':main()
