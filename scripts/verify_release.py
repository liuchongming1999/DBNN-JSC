"""Recompute released input hashes and DBNN metrics from per-input raw arrays."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
R=Path(__file__).resolve().parents[1];W=R/'workspace'
raw=json.loads((R/'docs/export_manifest.json').read_text())
extra=json.loads((R/'docs/additional_artifacts.json').read_text())
for f in raw['files']+extra['files']:
 p=R/f['path'];assert p.is_file(),f['path']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==f['sha256'],f['path']
manifest=json.loads((W/'benchmarks/spike_panel_20261009/main379/manifest.json').read_text())
train=json.loads((W/'benchmarks/accuracy_frontier_20260929/balanced_1ms/training_manifest.json').read_text())
old=raw['original_root'];training_hashes={r['input_sha256'] for r in train['cases']}
assert len(train['cases'])==149 and sum(r['split']=='fit' for r in train['cases'])==117
sys.path[:0]=[str(W/'benchmarks/training_refinement_20260929')]
from train_refinement import match_events
sse=0.;n=0;tp=fp=fn=spikes=0
for row in manifest['cases']:
 cid=row['case_id'];inp=W/Path(row['input_file']).relative_to(old)
 data=np.load(inp);assert hashlib.sha256(data['x'].tobytes()).hexdigest()==row['input_sha256']
 assert row['input_sha256'] not in training_hashes
 p=np.load(W/'benchmarks/random_init_main379_20261012/predictions'/(cid+'.npz'))
 t=p['time_ms'];ref=p['reference_voltage'].astype(float);v=p['voltage'].astype(float)
 ev=p['reference_events_ms'];keep=np.ones(len(t),bool)
 for s in ev:keep&=abs(t-float(s))>10.
 sse+=float(np.square(v[keep]-ref[keep]).sum());n+=int(keep.sum());spikes+=len(ev)
 score=match_events(ev,p['predicted_events_ms'],5.)
 tp+=score['tp'];fp+=score['fp'];fn+=score['fn']
metric={'cases':len(manifest['cases']),'reference_spikes':spikes,'pooled_rmse_mv':float(np.sqrt(sse/n)),
 'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp),'recall':tp/(tp+fn),'f1':2*tp/(2*tp+fp+fn)}
d=json.loads((R/'paper/evidence/main379_results.json').read_text())
expected=d['models']['gpu']['accuracy']
assert metric['cases']==379 and spikes==637
assert abs(metric['pooled_rmse_mv']-expected['pooled_rmse_mv'])<1e-9
for k in ['tp','fp','fn']:assert metric[k]==expected['events'][k],k
assert abs(metric['f1']-expected['events']['f1'])<1e-12
cp=W/'benchmarks/diagonal_1ms_20261009/dbnn_w500_dt1ms_burn150_full_diag/best_checkpoint.pt'
assert hashlib.sha256(cp.read_bytes()).hexdigest()==d['checkpoint_sha256']
print(json.dumps({'passed':True,'files_verified':len(raw['files'])+len(extra['files']),'input_split_disjoint':True,'raw_metrics':metric},indent=2))
