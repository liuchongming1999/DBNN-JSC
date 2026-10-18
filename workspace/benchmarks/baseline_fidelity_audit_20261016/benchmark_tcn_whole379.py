"""Fixed-weight audit: all379 TCN one-pass vs original overlap; paired DBNN.
Does not modify checkpoints, thresholds, paper results or existing predictions.
"""
from pathlib import Path
import sys,json,hashlib,os,time,gc
import torch
import numpy as np
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');B=R/'benchmarks/baselines_20260930';W=R/'benchmarks/baseline_fidelity_audit_20261016';O=W/'tcn_whole_main379'
sys.path[:0]=[str(B),str(R/'benchmarks/compute_fair_20260929'),str(R/'benchmarks/sparse_20260928'),str(R/'benchmarks/fair_window_20261005/timing_w500'),str(R/'benchmarks/solver_main379_20261016')]
from train_default import model_for,predict_case
from gpu_backend import ResolutionModel,Graph
from backends import subset_state
from event_thresholds import detected_events
from event_metrics import match_events,aggregate_events

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def metrics(v,ev,ref,re):
 mask=np.ones(len(v),bool);t=np.arange(len(v))
 for s in re:mask &= abs(t-s)>10
 return dict(sse=float(np.sum((v[mask].astype(float)-ref[mask].astype(float))**2)),n=int(mask.sum()),events=match_events(re,ev))
@torch.inference_mode()
def main():
 os.sched_setaffinity(0,{114});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 (O/'predictions').mkdir(parents=True,exist_ok=True);src=B/'aligned/tcn_1ms';lock=read(src/'complete.json');cfg=lock['config'];cal=lock['validation']['calibration'];th=read(src/'event_threshold.json')['threshold'];cp=src/'best.pt';assert sha(cp)==lock['checkpoint_sha256']
 model=model_for(B,cfg).cuda().eval();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model']);dcp=R/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt';dhash='737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d';assert sha(dcp)==dhash;state=torch.load(dcp,map_location='cuda',weights_only=False)['model_state']
 manifest=R/'benchmarks/solver_main379_20261016/manifest.json';rows=read(manifest)['cases'];policy=read(R/'benchmarks/fair_window_20261005/timing_w500/gpu_policy.json')['records'];cache={};records=[]
 write(O/'protocol.json',dict(checkpoint_sha256=sha(cp),dbnn_checkpoint_sha256=dhash,manifest_sha256=sha(manifest),cases=379,repeats=7,warmups=3,precision='FP32IEEE; TF32 disabled',physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),device=torch.cuda.get_device_name(),scope='Prepared B1 complete forward, CUDA Graph and synchronization for all methods; calibration/sigmoid included for TCN. Preparation, upload/download, capture and scoring excluded.',method_order='Fixed-seed randomized order per case',training_and_thresholds_unchanged=True))
 for idx,row in enumerate(rows):
  cid=row['case_id'];f=np.load(row['input_file']);raw=f['x'];ids=f['ids'];n=raw.shape[-1];assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256'];dense=np.zeros((n,500),np.float32);dense[:,ids]=raw.T
  if n not in cache:
   inp=torch.from_numpy(dense).cuda();pairs={}
   for flag,name in [(True,'tcn_author_windows'),(False,'tcn_whole')]:
    def forward(flag=flag,inp=inp):
     y=predict_case(model,inp,cfg,'cuda') if flag else model(inp[None])[0]
     return (y[:,1]-67.7)*cal['scale']+cal['bias_mv'],torch.sigmoid(y[:,0])
    for _ in range(3):out=forward()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=forward()
    pairs[name]=(g,out)
   cache[n]=(inp,pairs)
  inp,pairs=cache[n];inp.copy_(torch.from_numpy(dense));torch.cuda.synchronize()
  x=torch.from_numpy(raw.astype(np.int8)*np.where(ids[:,None]<250,1,-1).astype(np.int8))[None].contiguous().cuda();st=subset_state(state,torch.as_tensor(ids.astype(np.int64),device='cuda'));near=min([p for p in policy if p['dt_ms']==1. and p['A']==len(ids)],key=lambda p:abs(np.log(p['input_steps']/n)));mode,block=near['selected'].rsplit('_',1);db=ResolutionModel(st,x,1.,mode,int(block));dg=Graph(db)
  def fn_for(name):
   if name=='dbnn_frozen':return dg
   g,out=pairs[name]
   def fn():g.replay();return out
   return fn
  vals={};timing={};order=np.random.default_rng(161050+idx).permutation(['tcn_author_windows','tcn_whole','dbnn_frozen']).tolist()
  for name in order:
   fn=fn_for(name)
   for _ in range(3):out=fn()
   torch.cuda.synchronize();times=[]
   for _ in range(7):
    torch.cuda.synchronize();t=time.perf_counter();out=fn();torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
   timing[name]=dict(median_ms=float(np.median(times)),runs_ms=times);vals[name]=out[0].cpu().numpy().copy() if name=='dbnn_frozen' else [v.cpu().numpy().copy() for v in out]
  a,b=vals['tcn_author_windows'],vals['tcn_whole'];aev=detected_events(a[1],1.,th);bev=detected_events(b[1],1.,th);dev=np.flatnonzero(vals['dbnn_frozen']>=db.th).astype(float)
  old=np.load(R/'benchmarks/spike_panel_20261009/methods/tcn_author_default/predictions'/f'{cid}.npz');dv=np.load(R/'benchmarks/random_init_main379_20261012/predictions'/f'{cid}.npz')
  parity=dict(tcn_vs_canonical_max_mv=float(np.max(abs(a[0]-old['voltage']))),whole_voltage_max_mv=float(np.max(abs(a[0]-b[0]))),whole_probability_max=float(np.max(abs(a[1]-b[1]))),whole_events_equal=bool(np.array_equal(aev,bev)),dbnn_voltage_max_mv=float(np.max(abs(vals['dbnn_frozen']-dv['voltage']))),dbnn_events_equal=bool(np.array_equal(dev,dv['predicted_events_ms'])))
  assert parity['tcn_vs_canonical_max_mv']<1e-4
  ref=np.load(R/'benchmarks/solver_main379_20261016/outputs/reference1/dt0.0125'/f'{cid}.npz');mm={name:metrics(vals[name],dev,ref['voltage'],ref['spike_times_ms']) if name=='dbnn_frozen' else metrics(vals[name][0],aev if name=='tcn_author_windows' else bev,ref['voltage'],ref['spike_times_ms']) for name in vals}
  rec=dict(case_id=cid,duration_ms=row['duration_ms'],A=len(ids),parity=parity,timing=timing,metrics=mm,input_sha256=row['input_sha256']);records.append(rec)
  np.savez_compressed(O/'predictions'/f'{cid}.npz',voltage=b[0],probability=b[1],spike_times_ms=bev,dbnn_voltage=vals['dbnn_frozen'],dbnn_events_ms=dev)
  write(O/'status.json',dict(completed=idx+1,total=379));write(O/'records'/f'{cid}.json',rec)
  if (idx+1)%20==0:print('AUDIT_TCN_WHOLE',idx+1,379,flush=True)
  del db,dg,x,st,fn,out;gc.collect();torch.cuda.empty_cache()
 summary={name:dict(wall_ms=sum(r['timing'][name]['median_ms'] for r in records),rmse_mv=float(np.sqrt(sum(r['metrics'][name]['sse'] for r in records)/sum(r['metrics'][name]['n'] for r in records))),events=aggregate_events([r['metrics'][name]['events'] for r in records])) for name in ['tcn_author_windows','tcn_whole','dbnn_frozen']}
 parity_summary=dict(cases=379,whole_voltage_max_mv=max(r['parity']['whole_voltage_max_mv'] for r in records),whole_probability_max=max(r['parity']['whole_probability_max'] for r in records),whole_events_equal_cases=sum(r['parity']['whole_events_equal'] for r in records),dbnn_max_mv=max(r['parity']['dbnn_voltage_max_mv'] for r in records),dbnn_events_equal_cases=sum(r['parity']['dbnn_events_equal'] for r in records))
 write(O/'complete.json',dict(finished=True,summary=summary,parity=parity_summary,records=records,source_sha256=sha(__file__),checkpoints_unchanged=sha(cp)==lock['checkpoint_sha256'] and sha(dcp)==dhash));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
