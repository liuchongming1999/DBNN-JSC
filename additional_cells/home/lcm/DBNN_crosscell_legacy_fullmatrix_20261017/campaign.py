"""Finite replay of the legacy event-training stage; validation only."""
from pathlib import Path
import concurrent.futures as cf
import fcntl,json,math,os,queue,subprocess,sys,threading,time,csv
from common8 import read,write,sha
p=Path(__file__).resolve().parent
old=Path('/home/lcm/DBNN_spike_training_20261004')
root=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
guard=open(p/'campaign.lock','w');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert read(p/'preflight.json')['passed']
if not (p/'locked_protocol.json').exists():
 jobs=[]
 priority=['human_l23','ca1','hay_l5','granule','basket','golgi']
 for cell in priority:
  project=Path('/home/lcm')/('DBNN_multineuron_20261002' if cell in ['granule','basket','golgi'] else 'DBNN_noncerebellar_20261002')
  for profile in ['ampa','ampa_nmda']:
   oldout=old/'results'/cell/profile/'dt1.0ms';pr=read(oldout/'protocol.json');base=project/'results'/cell/profile/'dt1.0ms';m=base/'training_manifest.json'
   assert sha(base/'best_checkpoint.pt')==pr['original_checkpoint_sha256']
   assert sha(m)==pr['input_manifest_sha256']
   jobs.append(dict(cell=cell,profile=profile,project=str(project),old_output=str(oldout),starting_checkpoint=str(base/'best_checkpoint.pt'),starting_checkpoint_sha256=sha(base/'best_checkpoint.pt'),manifest=str(m),manifest_sha256=sha(m),fit=pr['fit_cases'],validation=pr['validation_cases'],expected_updates=3*1400*math.ceil(pr['fit_cases']/8)))
 lock=dict(created=time.strftime('%Y-%m-%dT%H:%M:%S%z'),purpose='Replay exact legacy event-training stage, freeing the diagonal only; the identical upstream voltage-fit checkpoint is shared, not retrained',diagonal_lr_multiplier=.1,diagonal_initialization='zero',seed=310049,trials=[0.,.2,1.],epochs=[1000,400],base_lrs=[.0003,.00006],gpus=[2,5],source_old_script_sha256=sha(old/'train_events.py'),script_sha256=sha(p/'train_events.py'),drive_script_sha256=sha(p/'full_matrix_drive.py'),input_policy='Exact archived training/validation manifests with original retained prefixes; no new output selection',test_policy='No test evaluation; no test-based version selection',main_model_unchanged=True,manuscript_unchanged=True,jobs=jobs)
 write(p/'locked_protocol.json',lock)
lock=read(p/'locked_protocol.json');assert lock['script_sha256']==sha(p/'train_events.py') and lock['drive_script_sha256']==sha(p/'full_matrix_drive.py')
q=queue.Queue()
for i,j in enumerate(lock['jobs']):q.put((i,j))
state=dict(status='running',pid=os.getpid(),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),jobs={},errors=[],validation_only=True)
mutex=threading.Lock()
def persist():state['updated']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write(p/'campaign_state.json',state)
def worker(gpu):
 while True:
  try:i,j=q.get_nowait()
  except queue.Empty:return
  assert sha(j['starting_checkpoint'])==j['starting_checkpoint_sha256'];assert sha(j['manifest'])==j['manifest_sha256']
  out=p/'results'/j['cell']/j['profile'];out.mkdir(parents=True,exist_ok=True)
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib')
  with mutex:state['jobs'][str(i)]=dict(status='running',cell=j['cell'],profile=j['profile'],gpu=gpu);persist()
  cmd=['taskset','-c',str(80+gpu),sys.executable,'-u',str(p/'train_events.py'),'--root',str(root),'--project',j['project'],'--work',str(p),'--cell',j['cell'],'--profile',j['profile'],'--dt','1','--epochs','1000','--refinement-epochs','400','--trials','0,0.2,1.0','--diagonal-lr-multiplier','.1']
  with (out/'training.log').open('a') as log:rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  if rc==0:
   s=read(out/'dt1.0ms/final_selection.json');assert s['updates']==j['expected_updates'] and not s['test_evaluated']
  with mutex:
   state['jobs'][str(i)].update(status='complete' if rc==0 else 'failed',returncode=rc)
   if rc:state['errors'].append(dict(job=i,returncode=rc))
   persist()
  if rc:return
# Explicitly reserve the two idle devices; do not use ELM or other training GPUs.
for _ in range(3):
 snapshot={int(r[0]):(int(r[1]),int(r[2])) for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).splitlines())}
 assert all(snapshot[g][0]<=5 and snapshot[g][1]<2048 for g in [2,5]),snapshot
 time.sleep(1)
with cf.ThreadPoolExecutor(max_workers=2) as pool:
 fs=[pool.submit(worker,g) for g in [2,5]]
 for f in fs:
  try:f.result()
  except Exception as e:
   with mutex:state['errors'].append(dict(error=repr(e)));persist()
rows=[]
for j in lock['jobs']:
 s=p/'results'/j['cell']/j['profile']/'dt1.0ms/final_selection.json'
 if not s.exists():continue
 a=read(Path(j['old_output'])/'final_selection.json');b=read(s);va=a['selected']['validation'];vb=b['selected']['validation']
 rows.append(dict(cell=j['cell'],profile=j['profile'],old=dict(rmse=va['pooled_rmse_mv'],f1=va['events']['f1'],score=va['score'],candidate=a['selected']['name']),new=dict(rmse=vb['pooled_rmse_mv'],f1=vb['events']['f1'],score=vb['score'],candidate=b['selected']['name']),diagonal_max_abs=b['diagonal_max_abs'],updates=b['updates'],checkpoint=str(s.with_name('best_checkpoint.pt')),checkpoint_sha256=b['checkpoint_sha256'],elapsed_s=b['elapsed_s']))
write(p/'validation_comparison.json',dict(complete=len(rows)==12,validation_only=True,test_evaluated=False,legacy_metrics_note='Old validation scores are archived; new event matching includes the already approved 1e-6 ms boundary tolerance. Baseline replay records any count correction.',rows=rows))
state['status']='complete' if len(rows)==12 and not state['errors'] else 'incomplete';state['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z');persist()
