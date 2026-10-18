"""Single dependent pipeline; no recurring remote jobs or training variants."""
from pathlib import Path
import time,os,json,subprocess,sys,traceback
W=Path(__file__).resolve().parent

def state(d):
 p=W/'campaign_state.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def main():
 pid=int((W/'training.pid').read_text());done=W/'aligned/elm_1ms/complete.json';state(dict(status='training_running',training_pid=pid))
 while not done.exists():
  p=Path(f'/proc/{pid}/stat')
  if not p.exists() or p.read_text().split()[2]=='Z':raise RuntimeError('Training process exited before full-budget completion')
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode();assert str(W/'train_author_aligned_elm.py') in cmd
  time.sleep(30)
 for stage,script in [('validation_and_379_evaluation','evaluate_elm.py'),('paired_final_gpu_timing','final_gpu_timing.py'),('collecting_ready_results','collect_ready.py')]:
  state(dict(status=stage))
  with (W/(stage+'.log')).open('w') as f:
   rc=subprocess.call([sys.executable,'-u',str(W/script)],stdout=f,stderr=subprocess.STDOUT)
  if rc:raise RuntimeError(f'{script} failed, returncode={rc}')
 state(dict(status='results_ready_manuscript_not_yet_updated',finished=True,ready_file=str(W/'comparison_ready.json')))
if __name__=='__main__':
 try:main()
 except Exception as e:
  state(dict(status='failed',error=repr(e),traceback=traceback.format_exc()));raise
