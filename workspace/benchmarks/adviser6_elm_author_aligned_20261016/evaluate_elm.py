"""Run only after full training; lock calibration on original validation32,
then evaluate corrected/old ELM and frozen DBNN on unchanged main379.
"""
from pathlib import Path
import sys,json,time,hashlib,os,gc
import torch
import numpy as np
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');W=Path(__file__).resolve().parent;B=R/'benchmarks/baselines_20260930'
sys.path[:0]=[str(W),str(B),str(R/'benchmarks/solver_main379_20261016')]
from train_author_aligned_elm import trainer
from panel import rows_for,make_case,save,digest
from event_thresholds import fit_threshold,detected_events
from event_metrics import match_events,aggregate_events
from train_default import predict_case

def score(v,ev,ref,refev):
 t=np.arange(len(v),dtype=float);mask=np.ones(len(v),dtype=bool)
 for e in refev:mask &= np.abs(t-e)>10
 err=v.astype(float)-ref.astype(float)
 return dict(sse=float(np.sum(err[mask]**2)),samples=int(mask.sum()),events=match_events(refev,ev))
def agg(values):return dict(rmse_mv=float(np.sqrt(sum(v['sse'] for v in values)/sum(v['samples'] for v in values))),events=aggregate_events([v['events'] for v in values]))
@torch.inference_mode()
def main():
 os.sched_setaffinity(0,{112});torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 src=W/'aligned/elm_1ms';lock=json.loads((src/'complete.json').read_text());cp=src/'best.pt';assert lock['default_budget_completed'] and lock['updates']==350000 and digest(cp)==lock['checkpoint_sha256']
 cfg=lock['config'];assert cfg['morphology_routing_override']['sha256']==digest(W/'morphology_mapping.json');model=trainer.model_for(W,cfg).cuda().eval();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model'])
 route=np.asarray(json.loads((W/'morphology_mapping.json').read_text())['routing']).ravel();assert np.array_equal(model.input_to_synapse_indices.cpu().numpy().ravel(),route) and bool(model.valid_indices_mask.all())
 out=W/'elm_evaluation';(out/'predictions').mkdir(parents=True,exist_ok=True)
 validation=[];refs=[];rows,_=rows_for(R,1.)
 for r in rows:
  if r['split']!='validation':continue
  c=make_case(r,1.,signed=True);y=predict_case(model,torch.from_numpy(c['x']).cuda(),cfg,'cuda');pr=torch.sigmoid(y[:,0]).cpu().numpy();validation.append(pr);refs.append(c['spike_times']);vp=out/'validation_predictions';vp.mkdir(exist_ok=True);np.savez_compressed(vp/f'{r["case_id"]}.npz',probability=pr,reference_spikes_ms=c['spike_times'])
 threshold=fit_threshold(validation,refs,1.,match_events,aggregate_events);threshold.update(checkpoint_sha256=digest(cp),source='Original32 validation trajectories only; unchanged threshold candidate rule, corrected inclusive5ms floating boundary')
 save(src/'event_threshold.json',threshold);cal=lock['validation']['calibration'];manifest=R/'benchmarks/solver_main379_20261016/manifest.json';rows=json.loads(manifest.read_text())['cases'];assert len(rows)==379
 oldsrc=B/'aligned/elm_1ms';oldlock=json.loads((oldsrc/'complete.json').read_text());assert digest(oldsrc/'best.pt')==oldlock['checkpoint_sha256']
 oldmodel=trainer.model_for(W,oldlock['config']).cuda().eval();oldmodel.load_state_dict(torch.load(oldsrc/'best.pt',map_location='cpu',weights_only=False)['model']);oldcal=oldlock['validation']['calibration'];oldthreshold=json.loads((oldsrc/'event_threshold.json').read_text())['threshold']
 caches={};records=[]
 for i,r in enumerate(rows):
  cid=r['case_id'];f=np.load(r['input_file']);raw=f['x'];ids=f['ids'];assert hashlib.sha256(raw.tobytes()).hexdigest()==r['input_sha256'];n=raw.shape[-1]
  dense=np.zeros((n,500),np.float32);dense[:,ids]=raw.T;dense[:,250:]*=-1
  results={}
  for method,mod,cf,ca,th in [('elm_corrected',model,cfg,cal,threshold['threshold']),('elm_original',oldmodel,oldlock['config'],oldcal,oldthreshold)]:
   key=(method,n)
   if key not in caches:
    inp=torch.from_numpy(dense).cuda()
    def forward(inp=inp,mod=mod,cf=cf,ca=ca):
     y=predict_case(mod,inp,cf,'cuda');return (y[:,1]*10-67.7)*ca['scale']+ca['bias_mv'],torch.sigmoid(y[:,0])
    for _ in range(3):eager=forward()
    torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):captured=forward()
    graph.replay();torch.cuda.synchronize();assert max(float(torch.max(torch.abs(a-b))) for a,b in zip(eager,captured))<1e-4
    caches[key]=(inp,graph,captured)
   inp,graph,captured=caches[key];inp.copy_(torch.from_numpy(dense));graph.replay();torch.cuda.synchronize();v,pr=[a.cpu().numpy().copy() for a in captured];ev=detected_events(pr,1.,th);times=[]
   for _ in range(3):
    torch.cuda.synchronize();start=time.perf_counter();graph.replay();torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
   if method=='elm_corrected':np.savez_compressed(out/'predictions'/f'{cid}.npz',voltage=v,probability=pr,spike_times=ev,time_ms=np.arange(n))
   else:
    old=np.load(R/'benchmarks/spike_panel_20261009/methods/elm_author_default/predictions'/f'{cid}.npz');assert np.max(np.abs(old['voltage']-v))<1e-4
   ref=np.load(R/'benchmarks/solver_main379_20261016/outputs/reference1/dt0.0125'/f'{cid}.npz');results[method]=dict(**score(v,ev,ref['voltage'],ref['spike_times_ms']),wall_median_ms=float(np.median(times)),runs_ms=times)
  db=np.load(R/'benchmarks/random_init_main379_20261012/predictions'/f'{cid}.npz');results['dbnn_frozen']=score(db['voltage'],db['predicted_events_ms'],ref['voltage'],ref['spike_times_ms'])
  records.append(dict(case_id=cid,family=r['family'],duration_ms=r['duration_ms'],results=results));save(out/'status.json',dict(completed=i+1,total=379))
  if (i+1)%20==0:print('ELM_EVAL',i+1,379,flush=True)
 summaries={m:dict(**agg([r['results'][m] for r in records]),**({'prepared_wall_ms':sum(r['results'][m]['wall_median_ms'] for r in records)} if m!='dbnn_frozen' else {})) for m in ['elm_corrected','elm_original','dbnn_frozen']}
 groups={fam:{m:agg([r['results'][m] for r in records if r['family']==fam]) for m in summaries} for fam in sorted({r['family'] for r in records})}
 save(out/'complete.json',dict(finished=True,summary=summaries,by_family=groups,records=records,corrected_checkpoint_sha256=digest(cp),original_checkpoint_sha256=oldlock['checkpoint_sha256'],main_manifest_sha256=digest(manifest),threshold=threshold['threshold'],timing='PreparedB1FP32IEEE graph replay on sameGPU;3rep medians; includes full author overlap inference and sigmoid; packing,transfers,capture and scoring excluded',model_selection='Full350000-update author budget; best original validation alignedRMSE only; event threshold validation only; no heldout selection',created=time.strftime('%Y-%m-%dT%H:%M:%S%z')));print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
