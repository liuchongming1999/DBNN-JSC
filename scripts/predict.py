"""One complete prediction with the paper checkpoint and recorded implementations."""
import argparse,sys,json,time
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--case',default='confirmatory_0532');p.add_argument('--backend',choices=['gpu','cpu'],default='gpu');p.add_argument('--threads',type=int,default=1);p.add_argument('--output',type=Path,default=R/'build/prediction.npz');p.add_argument('--repeats',type=int,default=3);a=p.parse_args()
w=a.workspace.resolve();cp=w/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt'
torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
sys.path[:0]=[str(w/'benchmarks/compute_fair_20260929'),str(w/'benchmarks/sparse_20260928'),str(w/'benchmarks/fair_window_20261005/timing_w500'),str(w/'benchmarks/cpu_20260929')]
state=torch.load(cp,map_location='cpu',weights_only=False)['model_state']
panel=w/'benchmarks/spike_panel_20261009'
data=np.load(panel/'confirmatory/inputs'/(a.case+'.npz'));ids=data['ids'].astype(np.int64);raw=data['x'];signed=raw.astype(np.float32)*np.where(ids[:,None]<250,1.,-1.)
ref=np.load(w/'benchmarks/random_init_main379_20261012/predictions'/(a.case+'.npz'))
with torch.inference_mode():
 if a.backend=='gpu':
  from gpu_backend import ResolutionModel,Graph
  from backends import subset_state
  ix=torch.from_numpy(ids).cuda();st=subset_state({k:v.cuda() for k,v in state.items()},ix)
  x=torch.from_numpy(signed.astype(np.int8))[None].contiguous().cuda()
  # Same policy as the paper, with numerical checks against the frozen oracle.
  policy=json.loads((w/'benchmarks/fair_window_20261005/timing_w500/gpu_policy.json').read_text())['records']
  candidates=[r for r in policy if r['dt_ms']==1. and r['A']==len(ids)]
  near=min(candidates,key=lambda r:abs(np.log(r['input_steps']/x.shape[-1])))
  for tag in sorted(near['timing'],key=lambda k:near['timing'][k]['median_ms']):
   mode,block=tag.rsplit('_',1);model=ResolutionModel(st,x,1.,mode,int(block));got=model()[0].cpu().numpy()
   if np.max(np.abs(got-ref['voltage']))<.001 and np.array_equal(np.flatnonzero(got>=model.th),ref['predicted_events_ms']):break
  else:raise RuntimeError('No equivalent GPU implementation passed.')
  graph=Graph(model);sync=torch.cuda.synchronize
  run=lambda:graph();out=lambda:model.out[0].cpu().numpy().copy();threshold=model.th
 else:
  from cpu_backend import CPUModel
  from benchmark_cpu import methods
  model=CPUModel(state,ids,signed.shape[1]);x=np.zeros((500,signed.shape[1]),np.float32);x[ids]=signed
  file='cpu8' if a.threads==8 else 'cpu1'
  policy=json.loads((w/f'benchmarks/fair_window_20261005/timing_w500/{file}/policy.json').read_text())['records']
  near=min([r for r in policy if r['dt_ms']==1. and r['A']==len(ids)],key=lambda r:abs(np.log(r['T']/x.shape[-1])))
  for tag in sorted(near['timing'],key=lambda k:near['timing'][k]['wall']['median_ms']):
   cfg=methods()[tag];got=model.run(x,a.threads,**cfg)
   if np.max(np.abs(got-ref['voltage']))<.001 and np.array_equal(np.flatnonzero(got>=model.threshold),ref['predicted_events_ms']):break
  else:raise RuntimeError('No equivalent CPU implementation passed.')
  run=lambda:model.run(x,a.threads,**cfg);sync=lambda:None;out=lambda:model.output.copy();threshold=model.threshold
 for _ in range(3):run()
 ms=[]
 for _ in range(a.repeats):
  sync();start=time.perf_counter();run();sync();ms.append((time.perf_counter()-start)*1000)
 v=out();ev=np.flatnonzero(v>=threshold).astype(float)
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,voltage=v,time_ms=np.arange(len(v)),predicted_events_ms=ev)
 gap=float(np.max(np.abs(v-ref['voltage'])));assert gap<.001 and np.array_equal(ev,ref['predicted_events_ms'])
 print(json.dumps({'case':a.case,'backend':a.backend,'threads':a.threads,'wall_median_ms':float(np.median(ms)),'runs_ms':ms,'max_difference_mv':gap,'events_identical':True,'output':str(a.output)}))
