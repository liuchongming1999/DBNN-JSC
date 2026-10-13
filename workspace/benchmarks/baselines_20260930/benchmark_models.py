"""Locked learned baselines: common main31 voltage/event metrics and full forward.

Threshold and moment alignment are fitted only to validation. Eager / CUDA Graph
implementations must produce the same voltage and event decisions. Publication
timings can be refreshed separately, after the training workers have finished.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from scipy.signal import find_peaks
from panel import rows_for,make_case,save,digest
from train_default import model_for,predict_case
from job_lock import acquire_lock
from event_thresholds import VERSION as THRESHOLD_VERSION,fit_threshold

def events(prob,dt,threshold):
    peaks,_=find_peaks(prob,height=threshold,distance=max(1,round(5/dt)))
    return peaks*dt

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--method',choices=['tcn','elm','lstm'],required=True);p.add_argument('--dt',type=float,required=True);p.add_argument('--repeats',type=int,default=30);p.add_argument('--publication',action='store_true');a=p.parse_args();root=a.root.resolve();work=a.work.resolve()
    src=work/'aligned'/f'{a.method}_{a.dt:g}ms';lock=json.loads((src/'complete.json').read_text());assert lock['default_budget_completed'] and lock['updates']==lock['total_updates']
    out=src/('publication' if a.publication else 'evaluation');out.mkdir(exist_ok=True)
    eval_guard=acquire_lock(out/'evaluation.lock')
    if not a.publication and (out/'complete.json').exists():
        previous=json.loads((out/'complete.json').read_text())
        if previous['checkpoint']['checkpoint_sha256']==lock['checkpoint_sha256'] and previous['runtime']['repeats']>=a.repeats and previous['runtime'].get('event_threshold_protocol')==THRESHOLD_VERSION:return
    # All initial baseline evaluations use CPU cores 72--79; serialize those
    # measurements when training moves onto several GPUs.
    cpu_guard=acquire_lock(work/'logs/cpu_evaluation.lock')
    assert digest(src/'best.pt')==lock['checkpoint_sha256'];cfg=lock['config'];cp=torch.load(src/'best.pt',map_location='cpu',weights_only=False)
    sys.path[:0]=[str(root/'benchmarks/resolution_frontier_20260929'),str(root/'benchmarks/training_refinement_20260929')]
    from common import metric,aggregate,raster
    from train_refinement import match_events,event_aggregate
    os.sched_setaffinity(0,{72});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=model_for(work,cfg).cuda().eval();model.load_state_dict(cp['model']);cal=lock['validation']['calibration']
    threshold_file=src/'event_threshold.json'
    saved_threshold=json.loads(threshold_file.read_text()) if threshold_file.exists() else {}
    if saved_threshold.get('version')==THRESHOLD_VERSION and saved_threshold.get('checkpoint_sha256')==lock['checkpoint_sha256']:th=saved_threshold['threshold']
    else:
        rows,_=rows_for(root,a.dt);pred=[];refs=[]
        with torch.inference_mode():
            for row in rows:
                if row['split']!='validation':continue
                c=make_case(row,a.dt,signed=a.method!='tcn');y=predict_case(model,torch.from_numpy(c['x']).cuda(),cfg,'cuda');pred.append(torch.sigmoid(y[:,0]).cpu().numpy());refs.append(c['spike_times'])
        selected=fit_threshold(pred,refs,a.dt,match_events,event_aggregate);th=selected['threshold'];save(threshold_file,dict(**selected,checkpoint_sha256=lock['checkpoint_sha256']))
    rows=json.loads((root/'benchmarks/dt_resolution_20260929/test/manifest.json').read_text())['cases'];records=[]
    runtime=dict(checkpoint_sha256=lock['checkpoint_sha256'],event_threshold_protocol=THRESHOLD_VERSION,shared_resources=True,timer='Complete prepared forward including overlap/burn-in stitching, voltage calibration and sigmoid output. GPU synchronization included; external packing, upload/download, graph capture and model load excluded. CPU1/CPU8 same scope.',repeats=a.repeats,precision='FP32 IEEE',cpu_affinity=list(os.sched_getaffinity(0)),device=torch.cuda.get_device_name(),gpu_usage_start=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv'],capture_output=True,text=True).stdout)
    for row in rows:
        cid=row['case_id'];d=np.load(row['input_file']);raw=d['x'];ids=d['ids'];assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256']
        x=np.zeros((round(row['duration_ms']/a.dt)+1,500),np.float32);x[:,ids]=raster(raw,a.dt).T
        if a.method!='tcn':x[:,250:]*=-1
        ref=np.load(root/'benchmarks/dt_resolution_20260929/test/references'/f'{cid}_reference.npz');voltage=ref['voltage'][::2];refevents=ref['spike_times'];rr={}
        with torch.inference_mode():
            def forward(m,inp):
                y=predict_case(m,inp,cfg,str(inp.device));v=y[:,1]*(10 if a.method!='tcn' else 1)-67.7
                return v*cal['scale']+cal['bias_mv'],torch.sigmoid(y[:,0]),v
            inp=torch.from_numpy(x).cuda();model.cuda();base=forward(model,inp);torch.cuda.synchronize();gpu_output=[v.cpu().numpy().copy() for v in base];graph=None;graph_reason=None
            try:
                for _ in range(3):base=forward(model,inp)
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):captured=forward(model,inp)
                graph.replay();torch.cuda.synchronize();err=max(float(np.max(abs(v.cpu().numpy()-b))) for v,b in zip(captured,gpu_output));assert err<1e-4,('Graph mismatch',err)
                assert np.array_equal(events(captured[1].cpu().numpy(),a.dt,th),events(gpu_output[1],a.dt,th))
            except Exception as e:graph=None;graph_reason=repr(e);torch.cuda.empty_cache()
            timings=[]
            for _ in range(a.repeats):
                torch.cuda.synchronize();s=time.perf_counter()
                if graph is not None:graph.replay()
                else:base=forward(model,inp)
                torch.cuda.synchronize();timings.append((time.perf_counter()-s)*1000)
            rr['gpu']=dict(wall_ms=float(np.median(timings)),runs_ms=timings,implementation='CUDA Graph' if graph is not None else 'Author inference, eager',capture_failure=graph_reason)
            mm=metric(voltage,gpu_output[0][::round(1/a.dt)],refevents,1);unadjusted=metric(voltage,gpu_output[2][::round(1/a.dt)],refevents,1);ee=match_events(refevents,events(gpu_output[1],a.dt,th))
            np.savez_compressed(out/f'{cid}.npz',voltage=gpu_output[0],raw_voltage=gpu_output[2],probability=gpu_output[1],spike_times=events(gpu_output[1],a.dt,th))
            del graph,base;model.cpu();del inp;torch.cuda.empty_cache();inp_cpu=torch.from_numpy(x)
            for threads in [1,8]:
                os.sched_setaffinity(0,set(range(72,72+threads)));torch.set_num_threads(threads)
                for _ in range(3):cy=forward(model,inp_cpu)
                cpu_v=cy[0].numpy();gap=float(np.max(abs(cpu_v-gpu_output[0])));assert gap<.002,('CPU/GPU numeric gap',cid,gap)
                timings=[]
                for _ in range(a.repeats):s=time.perf_counter();cy=forward(model,inp_cpu);timings.append((time.perf_counter()-s)*1000)
                cpum=metric(voltage,cpu_v[::round(1/a.dt)],refevents,1.);cpue=match_events(refevents,events(cy[1].numpy(),a.dt,th))
                rr[f'cpu{threads}']=dict(wall_ms=float(np.median(timings)),runs_ms=timings,maximum_voltage_gap_mv=gap,metrics=cpum,events=cpue,cpu_affinity=list(range(72,72+threads)))
            os.sched_setaffinity(0,{72});torch.set_num_threads(1);model.cuda()
        rec=dict(case_id=cid,A=len(ids),duration_s=row['duration_ms']/1000,input_sha256=row['input_sha256'],metrics=mm,raw_metrics=unadjusted,events=ee,backends=rr);records.append(rec);save(out/f'{cid}.json',rec);save(out/'status.json',dict(completed=len(records),total=31))
    summary=dict(metrics=aggregate([r['metrics'] for r in records]),raw_metrics=aggregate([r['raw_metrics'] for r in records]),events=event_aggregate([r['events'] for r in records]),wall_ms={b:sum(r['backends'][b]['wall_ms'] for r in records) for b in ['gpu','cpu1','cpu8']})
    save(out/'complete.json',dict(finished=True,method=a.method,dt_ms=a.dt,checkpoint=lock,threshold=th,summary=summary,records=records,runtime=runtime))

if __name__=='__main__':main()
