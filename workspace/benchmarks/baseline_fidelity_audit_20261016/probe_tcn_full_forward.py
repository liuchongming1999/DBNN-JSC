"""Frozen TCN: diagnostic of equivalent whole-trajectory vs author-window inference.
No training, no checkpoint/threshold selection, no replacement of paper timings.
"""
from pathlib import Path
import sys,json,os,time,gc,hashlib
import numpy as np
import torch
R=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');B=R/'benchmarks/baselines_20260930';W=R/'benchmarks/baseline_fidelity_audit_20261016'
sys.path[:0]=[str(B)]
from train_default import model_for,predict_case
from event_thresholds import detected_events
@torch.inference_mode()
def main():
 os.sched_setaffinity(0,{114});torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.set_float32_matmul_precision('highest')
 src=B/'aligned/tcn_1ms';lock=json.loads((src/'complete.json').read_text());cfg=lock['config'];cal=lock['validation']['calibration'];threshold=json.loads((src/'event_threshold.json').read_text())['threshold'];cp=src/'best.pt';assert hashlib.sha256(cp.read_bytes()).hexdigest()==lock['checkpoint_sha256']
 model=model_for(B,cfg).cuda().eval();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False)['model']);rows=json.loads((R/'benchmarks/solver_main379_20261016/manifest.json').read_text())['cases'];cases=[next(r for r in rows if r['duration_ms']==t and r['A']==500) for t in [1000,8000,16000]];results=[]
 for row in cases:
  f=np.load(row['input_file']);dense=np.zeros((f['x'].shape[1],500),np.float32);dense[:,f['ids']]=f['x'].T;x=torch.from_numpy(dense).cuda()
  def forward(window):
   y=predict_case(model,x,cfg,'cuda') if window else model(x[None])[0]
   return (y[:,1]-67.7)*cal['scale']+cal['bias_mv'],torch.sigmoid(y[:,0])
  outputs={};times={};graphs=[]
  for flag,name in [(True,'author_windows'),(False,'whole_trajectory')]:
   for _ in range(3):out=forward(flag)
   torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):out=forward(flag)
   g.replay();torch.cuda.synchronize();outputs[name]=[v.cpu().numpy().copy() for v in out];ms=[]
   for _ in range(20):
    torch.cuda.synchronize();start=time.perf_counter();g.replay();torch.cuda.synchronize();ms.append((time.perf_counter()-start)*1000)
   times[name]=dict(median_ms=float(np.median(ms)),runs_ms=ms);graphs.append(g)
  a,b=outputs['author_windows'],outputs['whole_trajectory'];rec=dict(case_id=row['case_id'],duration_ms=row['duration_ms'],max_voltage_gap_mv=float(np.max(abs(a[0]-b[0]))),max_probability_gap=float(np.max(abs(a[1]-b[1]))),events_equal=bool(np.array_equal(detected_events(a[1],1.,threshold),detected_events(b[1],1.,threshold))),timing=times,speed_ratio=times['author_windows']['median_ms']/times['whole_trajectory']['median_ms']);results.append(rec);print(rec,flush=True)
  del graphs,g,out,x;gc.collect();torch.cuda.empty_cache()
 (W/'tcn_whole_forward_probe.json').write_text(json.dumps(dict(finished=True,records=results,checkpoint_sha256=lock['checkpoint_sha256'],physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),scope='Three prespecified duration/shape cases, sameFP32 weights and locked calibration/threshold. Diagnostic only; not all379 re-evaluation or manuscript replacement.'),indent=2)+'\n')
if __name__=='__main__':main()
