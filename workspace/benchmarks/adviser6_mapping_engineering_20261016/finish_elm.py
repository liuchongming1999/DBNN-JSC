"""Dependent evaluation supervisor, not a recurring monitor."""
from pathlib import Path
import time,os,json,subprocess,sys
W=Path('/home/lcm/DBNN_L5PC_surrogate_20260928/benchmarks/adviser6_mapping_engineering_20261016')
pid=int((W/'elm_training.pid').read_text());done=W/'aligned/elm_1ms/complete.json'
while not done.exists():
 try:os.kill(pid,0)
 except ProcessLookupError:
  (W/'elm_campaign_state.json').write_text(json.dumps(dict(status='training_process_exited_before_completion',pid=pid)));sys.exit(1)
 # A detached child may be a zombie briefly; avoid treating it as running indefinitely.
 p=Path(f'/proc/{pid}/stat')
 if p.exists() and p.read_text().split()[2]=='Z':
  (W/'elm_campaign_state.json').write_text(json.dumps(dict(status='training_process_zombie_before_completion',pid=pid)));sys.exit(1)
 time.sleep(30)
(W/'elm_campaign_state.json').write_text(json.dumps(dict(status='evaluation_running')))
with (W/'elm_evaluation.log').open('w') as f:rc=subprocess.call([sys.executable,'-u',str(W/'evaluate_mapped_elm.py')],stdout=f,stderr=subprocess.STDOUT)
(W/'elm_campaign_state.json').write_text(json.dumps(dict(status='evaluation_complete' if rc==0 else 'evaluation_failed',returncode=rc)))
raise SystemExit(rc)
