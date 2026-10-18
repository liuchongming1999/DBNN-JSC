"""Final prepared GPU timing after corrected ELM, fixed weights, same7 reps.
TCN whole-trajectory; ELM/LSTM unchanged author overlap; DBNN unchanged.
"""
from pathlib import Path
import sys,json,hashlib,os,time,gc,argparse
import torch
import numpy as np
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');B=R/'benchmarks/baselines_20260930';W=Path(__file__).resolve().parent
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
 ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');args=ap.parse_args()
 os.sched_setaffinity(0,{112});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 O=W/('final_gpu_recheck_smoke_20261017' if args.smoke else 'final_gpu_recheck_20261017');(O/'predictions').mkdir(parents=True,exist_ok=True)
 models={};meta={}
 for name,method in [('tcn_whole','tcn'),('elm_corrected','elm'),('lstm_unchanged','lstm')]:
  src=(W if method=='elm' and not args.smoke else B)/'aligned'/f'{method}_1ms';lock=read(src/'complete.json');cfg=lock['config'];cal=lock['validation']['calibration'];threshold=read(src/'event_threshold.json')['threshold'];cp=src/'best.pt';assert sha(cp)==lock['checkpoint_sha256']
  m=model_for(B,cfg).cuda().eval();m.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'])
  models[name]=(m,cfg,cal,threshold);meta[name]=dict(checkpoint=str(cp),sha256=sha(cp),cfg=cfg)
 dcp=R/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt';dhash='737efcff1e6686bfb153fca8590a8b28a605fbdb08f9a68ff44e68af8867686d';assert sha(dcp)==dhash;state=torch.load(dcp,map_location='cuda',weights_only=False)['model_state'];meta['dbnn_frozen']=dict(checkpoint=str(dcp),sha256=dhash)
 manifest=R/'benchmarks/solver_main379_20261016/manifest.json';rows=read(manifest)['cases'];assert len(rows)==379
 if args.smoke:rows=[rows[0],next(r for r in rows if r['A']==64)]
 policy=read(R/'benchmarks/fair_window_20261005/timing_w500/gpu_policy.json')['records'];cache={};records=[]
 protocol=dict(models=meta,manifest_sha256=sha(manifest),cases=len(rows),full_results=not args.smoke,repeats=7,warmups=3,precision='FP32IEEE; TF32 disabled',physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),device=torch.cuda.get_device_name(),scope='Prepared B1 complete forward; includes calibration/sigmoid for learned comparators, and host synchronization. Preparation, packing, transfers, capture and scoring excluded.',method_order='Fixed-seed randomized order per input',all_training_and_thresholds_unchanged=True)
 write(O/'protocol.json',protocol)
 for idx,row in enumerate(rows):
  cid=row['case_id'];f=np.load(row['input_file']);raw=f['x'];ids=f['ids'];n=raw.shape[-1];assert hashlib.sha256(raw.tobytes()).hexdigest()==row['input_sha256'];dense=np.zeros((n,500),np.float32);dense[:,ids]=raw.T
  pairs={}
  for name,(model,cfg,cal,th) in models.items():
   dat=dense.copy()
   if name!='tcn_whole':dat[:,250:]*=-1
   key=(name,n)
   if key not in cache:
    inp=torch.from_numpy(dat).cuda()
    def forward(inp=inp,model=model,cfg=cfg,cal=cal,name=name):
     y=model(inp[None])[0] if name=='tcn_whole' else predict_case(model,inp,cfg,'cuda')
     v=y[:,1]*(1 if name=='tcn_whole' else 10)-67.7
     return v*cal['scale']+cal['bias_mv'],torch.sigmoid(y[:,0])
    for _ in range(3):out=forward()
    torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=forward()
    cache[key]=(inp,g,out)
   inp,g,out=cache[key];inp.copy_(torch.from_numpy(dat));pairs[name]=(g,out)
  torch.cuda.synchronize();x=torch.from_numpy(raw.astype(np.int8)*np.where(ids[:,None]<250,1,-1).astype(np.int8))[None].contiguous().cuda();st=subset_state(state,torch.as_tensor(ids.astype(np.int64),device='cuda'));near=min([p for p in policy if p['dt_ms']==1. and p['A']==len(ids)],key=lambda p:abs(np.log(p['input_steps']/n)));mode,block=near['selected'].rsplit('_',1);db=ResolutionModel(st,x,1.,mode,int(block));dg=Graph(db)
  vals={};timing={};order=np.random.default_rng(161600+idx).permutation(list(models)+['dbnn_frozen']).tolist()
  for name in order:
   if name=='dbnn_frozen':fn=dg
   else:
    g,out=pairs[name]
    def fn(g=g,out=out):g.replay();return out
   for _ in range(3):out=fn()
   torch.cuda.synchronize();times=[]
   for _ in range(7):
    torch.cuda.synchronize();t=time.perf_counter();out=fn();torch.cuda.synchronize();times.append((time.perf_counter()-t)*1000)
   timing[name]=dict(median_ms=float(np.median(times)),runs_ms=times);vals[name]=out[0].cpu().numpy().copy() if name=='dbnn_frozen' else [v.cpu().numpy().copy() for v in out]
  ref=np.load(R/'benchmarks/solver_main379_20261016/outputs/reference1/dt0.0125'/f'{cid}.npz');mm={};parity={};saved={}
  for name in vals:
   if name=='dbnn_frozen':
    v=vals[name];ev=np.flatnonzero(v>=db.th).astype(float);old=np.load(R/'benchmarks/random_init_main379_20261012/predictions'/f'{cid}.npz');gap=float(np.max(abs(v-old['voltage'])));events_equal=bool(np.array_equal(ev,old['predicted_events_ms']));prob_gap=0
   else:
    v,pr=vals[name];ev=detected_events(pr,1.,models[name][3]);folder={'tcn_whole':R/'benchmarks/spike_panel_20261009/methods/tcn_author_default/predictions','lstm_unchanged':R/'benchmarks/spike_panel_20261009/methods/lstm_common150/predictions','elm_corrected':(B.parent/'spike_panel_20261009/methods/elm_author_default/predictions') if args.smoke else W/'elm_evaluation/predictions'}[name]
    old=np.load(folder/f'{cid}.npz');gap=float(np.max(abs(v-old['voltage'])));prob_gap=float(np.max(abs(pr-old['probability'])));events_equal=bool(np.array_equal(ev,old['spike_times']));saved[name+'_probability']=pr
   parity[name]=dict(max_voltage_gap_mv=gap,max_probability_gap=prob_gap,events_equal=events_equal,passed=gap<.001 and prob_gap<.0001 and events_equal);assert parity[name]['passed'],(cid,name,parity[name]);assert np.isfinite(v).all()
   mm[name]=metrics(v,ev,ref['voltage'],ref['spike_times_ms']);saved[name+'_voltage']=v;saved[name+'_events_ms']=ev
  rec=dict(case_id=cid,duration_ms=row['duration_ms'],A=len(ids),parity=parity,timing=timing,metrics=mm,input_sha256=row['input_sha256']);records.append(rec);np.savez_compressed(O/'predictions'/f'{cid}.npz',**saved);write(O/'records'/f'{cid}.json',rec);write(O/'status.json',dict(completed=idx+1,total=len(rows)))
  if (idx+1)%20==0:print('FINAL_TIMING',idx+1,len(rows),flush=True)
  del db,dg,x,st,fn,out;gc.collect();torch.cuda.empty_cache()
 summary={name:dict(wall_ms=sum(r['timing'][name]['median_ms'] for r in records),rmse_mv=float(np.sqrt(sum(r['metrics'][name]['sse'] for r in records)/sum(r['metrics'][name]['n'] for r in records))),events=aggregate_events([r['metrics'][name]['events'] for r in records])) for name in list(models)+['dbnn_frozen']}
 write(O/'complete.json',dict(finished=True,full_results=not args.smoke,protocol=protocol,summary=summary,records=records,source_sha256=sha(__file__),checkpoints_unchanged=all(sha(Path(m['checkpoint']))==m['sha256'] for m in meta.values())));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
