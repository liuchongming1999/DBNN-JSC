"""Frozen full-matrix DBNN, cumulative implementation controls on all main379.
No model fitting, threshold changes, input selection or precision reduction.
"""
from pathlib import Path
import sys,json,hashlib,time,os,gc,argparse,subprocess
import numpy as np
import torch
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');W=R/'benchmarks/adviser6_mapping_engineering_20261016'
sys.path[:0]=[str(R/'benchmarks/compute_fair_20260929'),str(R/'benchmarks/sparse_20260928'),str(R/'benchmarks/fair_window_20261005/timing_w500'),str(R/'benchmarks/solver_main379_20261016')]
from gpu_backend import ResolutionModel,Graph
from backends import native,subset_state
from event_metrics import match_events,aggregate_events
CP=R/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt'; EXPECT='737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def stat(x):return dict(median_ms=float(np.median(x)),q1_ms=float(np.quantile(x,.25)),q3_ms=float(np.quantile(x,.75)),runs_ms=list(map(float,x)))
LABELS=['R0_no_packing','R1_serial_filter','R2_unfused_drive','R3_sequential_reset','R4_complete_graph','R5_no_graph']
class Rollback:
 def __init__(self,m,removed):
  self.m=m;self.removed=removed;self.full=m.x.float().contiguous();self.z=torch.empty_like(m.y);self.tmp=torch.empty_like(m.y);self.linear=torch.empty_like(m.drive);self.quad=torch.empty_like(m.drive)
 def __call__(self):
  m=self.m;stream=torch.cuda.current_stream().cuda_stream
  if self.removed=='filter':native().call('dbnn_scan',self.full,m.y,m.da,m.dc,m.om,m.B,m.K,m.T,stream)
  else:m.filter()
  if self.removed=='fusion':
   torch.mm(m.y.reshape(-1,m.K),m.W,out=self.z.reshape(-1,m.K));torch.mul(m.y,self.z,out=self.tmp)
   torch.sum(self.tmp,dim=-1,out=self.quad);torch.sum(m.y,dim=-1,out=self.linear)
   torch.add(self.quad,self.linear,out=m.drive);m.drive.add_(m.bias)
  else:m.integrate()
  if self.removed=='reset':native().call('dbnn_reset',m.drive,m.out,m.B,m.T,m.alpha,m.amp,m.th,torch.cuda.current_stream().cuda_stream)
  else:m.reset()
  return m.out
@torch.inference_mode()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=0);ap.add_argument('--repeats',type=int,default=30);a=ap.parse_args()
 os.sched_setaffinity(0,{113});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 out=W/('engineering_rollback_smoke' if a.limit else 'engineering_rollback');out.mkdir(exist_ok=True)
 assert sha(CP)==EXPECT;state=torch.load(CP,map_location='cuda',weights_only=False)['model_state'];assert torch.diagonal(state['bilinear.weight'][0]).abs().max()>0
 manifest=R/'benchmarks/solver_main379_20261016/manifest.json';rows=read(manifest)['cases'];assert len(rows)==379
 if a.limit:rows=[rows[0],next(r for r in rows if r['A']==64)][:a.limit]
 policy=read(R/'benchmarks/fair_window_20261005/timing_w500/gpu_policy.json')['records']
 protocol=dict(checkpoint=str(CP),checkpoint_sha256=EXPECT,main_manifest_sha256=sha(manifest),variants=LABELS,rows=len(rows),repeats=a.repeats,warmups=5,precision='FP32 IEEE; TF32 disabled; full quadratic matrix including trained diagonal retained',device=torch.cuda.get_device_name(),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),cpu_affinity=sorted(os.sched_getaffinity(0)),scope='Prepared B=1 complete forward. Includes host launch and synchronization; excludes model/shape preparation, active-ID packing, transfers, graph capture and scoring. GPU-device timers measured separately.',policy='Fixed preceding synthetic shape policy selected tag; no new timing or accuracy selection among kernels',timing_order='Seeded per-input variant permutation, same repeat budget',source_sha256=sha(__file__),interpretation='One-at-a-time removal from the complete implementation. All graph variants use the same graph capture wrapper; no new kernel search. Includes a repeated complete and no-graph control for paired timings.',started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
 write(out/'protocol.json',protocol);records=[]
 for i,row in enumerate(rows):
  cid=row['case_id'];dest=out/'cases'/f'{cid}.json'
  if dest.exists():
   rec=read(dest);assert rec['checkpoint_sha256']==EXPECT and rec['script_sha256']==sha(__file__);records.append(rec);continue
  with np.load(row['input_file']) as f:ids=f['ids'].astype(np.int64);raw=f['x'];assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256']
  signed=raw.astype(np.int8)*np.where(ids[:,None]<250,1,-1).astype(np.int8);x=torch.from_numpy(signed)[None].contiguous().cuda();dense=torch.zeros((1,500,x.shape[-1]),dtype=torch.int8,device='cuda');dense[:,torch.as_tensor(ids,device='cuda')]=x
  st=subset_state(state,torch.as_tensor(ids,device='cuda')); near=min([p for p in policy if p['dt_ms']==1. and p['A']==len(ids)],key=lambda p:abs(np.log(p['input_steps']/x.shape[-1])));tag=near['selected'];mode,block=tag.rsplit('_',1)
  packed=ResolutionModel(st,x,1.,mode,int(block));allm=ResolutionModel(state,dense,1.,mode,int(block))
  backs=[Rollback(allm,'packing'),Rollback(packed,'filter'),Rollback(packed,'fusion'),Rollback(packed,'reset'),Rollback(packed,'none')];variants=[Graph(q) for q in backs]+[Rollback(packed,'none')];graph=variants[4]
  with np.load(R/'benchmarks/random_init_main379_20261012/predictions'/f'{cid}.npz') as f:oracle=f['voltage'];ev0=f['predicted_events_ms']
  with np.load(R/'benchmarks/solver_main379_20261016/outputs/reference1/dt0.0125'/f'{cid}.npz') as f:ref=f['voltage'].astype(float);refev=f['spike_times_ms']
  grid=np.arange(len(ref),dtype=float);mask=np.ones(len(ref),dtype=bool)
  for e in refev:mask &= np.abs(grid-e)>10
  result={};saved={}
  order=np.random.default_rng(161006+i).permutation(6).tolist()
  for j in order:
   fn=variants[j]
   for _ in range(5):fn()
   torch.cuda.synchronize();v=fn()[0].cpu().numpy().copy();events=np.flatnonzero(v>=packed.th).astype(float)
   gap=v.astype(float)-oracle;finite=bool(np.isfinite(v).all());assert finite
   parity=dict(max_abs_mv=float(np.max(np.abs(gap))),rmse_mv=float(np.sqrt(np.mean(gap**2))),events_equal=bool(np.array_equal(events,ev0)),finite=finite)
   parity['passed']=parity['max_abs_mv']<.001 and parity['events_equal'];wall=[];dev=[]
   for _ in range(a.repeats):
    torch.cuda.synchronize();t=time.perf_counter();fn();torch.cuda.synchronize();wall.append((time.perf_counter()-t)*1000)
   e0=torch.cuda.Event(enable_timing=True);e1=torch.cuda.Event(enable_timing=True)
   for _ in range(a.repeats):
    e0.record();fn();e1.record();e1.synchronize();dev.append(e0.elapsed_time(e1))
   err=v.astype(float)-ref;metrics=dict(sse=float(np.sum(err[mask]**2)),samples=int(mask.sum()),events=match_events(refev,events))
   result[LABELS[j]]=dict(wall=stat(wall),device=stat(dev),parity=parity,metrics=metrics)
   saved[LABELS[j]+'_voltage']=v;saved[LABELS[j]+'_events_ms']=events
  np.savez_compressed(out/'cases'/f'{cid}.npz',**saved) if (out/'cases').exists() else None
  rec=dict(case_id=cid,A=len(ids),duration_ms=row['duration_ms'],input_sha256=row['input_sha256'],checkpoint_sha256=EXPECT,script_sha256=sha(__file__),parallel_kernel=tag,variant_order=order,results=result)
  write(dest,rec)
  # Save predictions after JSON writer creates the cases directory on first row.
  np.savez_compressed(out/'cases'/f'{cid}.npz',**saved)
  records.append(rec);write(out/'status.json',dict(completed=i+1,total=len(rows),updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),parity_failures={k:sum(not r['results'][k]['parity']['passed'] for r in records) for k in LABELS}))
  if (i+1)%20==0:print('ENGINEERING',i+1,len(rows),flush=True)
  del variants,backs,graph,packed,allm,x,dense,st,fn;gc.collect();torch.cuda.empty_cache()
 summary={}
 for k in LABELS:
  vals=[r['results'][k] for r in records];summary[k]=dict(wall_ms=sum(v['wall']['median_ms'] for v in vals),device_ms=sum(v['device']['median_ms'] for v in vals),parity_passed=sum(v['parity']['passed'] for v in vals),cases=len(vals),max_abs_mv=max(v['parity']['max_abs_mv'] for v in vals),rmse_mv=float(np.sqrt(sum(v['metrics']['sse'] for v in vals)/sum(v['metrics']['samples'] for v in vals))),events=aggregate_events([v['metrics']['events'] for v in vals]))
 write(out/'complete.json',dict(finished=True,protocol=protocol,summary=summary,records=records,checkpoint_unchanged=sha(CP)==EXPECT,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z')));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
