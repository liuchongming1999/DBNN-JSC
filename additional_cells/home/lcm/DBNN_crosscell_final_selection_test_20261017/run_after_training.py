"""Finite dependency pipeline: freeze all models, then run the entire locked panel."""
from pathlib import Path
import concurrent.futures as cf
import csv,fcntl,json,os,queue,shutil,subprocess,sys,threading,time
import torch
from common8 import read,write,sha
p=Path(__file__).resolve().parent;guard=open(p/'campaign.lock','w');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
lock=read(p/'locked_protocol.json');training=Path(lock['training_work']);jobs=lock['jobs']
state=read(p/'campaign_state.json') if (p/'campaign_state.json').exists() else dict(jobs={},errors=[])
state.update(status='waiting_training',pid=os.getpid(),started=state.get('started',time.strftime('%Y-%m-%dT%H:%M:%S%z')),input_list_locked=True,publication_update=False)
mutex=threading.Lock()
def persist():state['updated']=time.strftime('%Y-%m-%dT%H:%M:%S%z');write(p/'campaign_state.json',state)
persist()
def on_error(kind,value,tb):
 state['status']='failed';state['errors'].append({'error':repr(value)});persist();sys.__excepthook__(kind,value,tb)
sys.excepthook=on_error
# All twelve trained/validation-selected models must exist before any test call.
while True:
 s=read(training/'campaign_state.json');n=sum(v['status']=='complete' for v in s['jobs'].values());state['training_completed']=n;persist()
 if s['status'] in ['incomplete','failed'] or s.get('errors'):raise RuntimeError(('training_requires_attention',s))
 if s['status']=='complete':assert n==12;break
 time.sleep(30)
assert sha(training/'locked_protocol.json')==lock['training_protocol_sha256']
assert read(training/'verification.json')['passed']
assert len(read(training/'verification.json')['conditions'])==12
if not (p/'models_locked.json').exists():
 models={}
 for i,j in enumerate(jobs):
  folder=Path(j['training_folder']);selection=read(folder/'final_selection.json');cp=folder/'best_checkpoint.pt';pr=selection['protocol']
  assert selection['finished'] and not selection['test_evaluated'] and selection['updates']==selection['expected_updates']==j['expected_updates']
  assert pr['diagonal_optimizer_group'] and pr['diagonal_lr_multiplier']==.1 and not pr['test_used']
  assert sha(cp)==selection['checkpoint_sha256'];assert sha(j['test_manifest'])==j['test_manifest_sha256'];assert sha(j['reference_completion'])==j['reference_completion_sha256']
  out=Path(j['base']);dest=out/'model.pt';shutil.copy2(cp,dest);assert sha(dest)==selection['checkpoint_sha256']
  chosen=torch.load(dest,map_location='cpu',weights_only=False);st=chosen['model_state'];diag=st['bilinear.weight'][0].diagonal()
  assert st['bilinear.weight'].shape==(1,500,500) and all(z.isfinite().all() for z in st.values())
  for k in ['tau_rise','tau_decay','tau_reset']:assert (st[k]>0).all()
  models[str(i)]=dict(cell=j['cell'],profile=j['profile'],checkpoint=str(dest),checkpoint_sha256=sha(dest),source_checkpoint=str(cp),training_selection_sha256=sha(folder/'final_selection.json'),selected_candidate=selection['selected']['name'],validation=selection['selected']['validation'],updates=selection['updates'],diagonal_trainable=True,diagonal_max_abs=float(diag.abs().max()),diagonal_nonzero=int(diag.count_nonzero()),reset={k:float(st[k]) for k in ['v_th','reset_amp','tau_reset']},model_locked_at=selection['locked_at'])
 write(p/'models_locked.json',dict(all_models_locked=True,locked_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),before_first_test=True,models=models,evaluate_script_sha256=sha(p/'evaluate.py'),precision_fallback_sha256=sha(p/'precision_fallback.py'),test_protocol_sha256=sha(p/'locked_protocol.json')))
models=read(p/'models_locked.json');assert models['all_models_locked'] and len(models['models'])==12
assert models['evaluate_script_sha256']==sha(p/'evaluate.py')
assert models['precision_fallback_sha256']==sha(p/'precision_fallback.py')
# Wait for the two formerly used training devices, never take ELM/other devices.
def idle():
 g={int(r[0]):(int(r[1]),int(r[2]),r[3].strip()) for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,uuid','--format=csv,noheader,nounits'],text=True).splitlines())}
 uu={g[k][2]:k for k in [2,5]}
 for r in csv.reader(subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True).splitlines()):
  if len(r)!=2 or r[0].strip() not in uu:continue
  try:cmd=Path('/proc',r[1].strip(),'cmdline').read_bytes().decode().replace('\0',' ')
  except FileNotFoundError:continue
  except PermissionError:return False
  if 'soffice' not in cmd.lower() and 'libreoffice' not in cmd.lower():return False
 return all(g[k][0]<=5 and g[k][1]<2048 for k in [2,5])
state.update(status='waiting_gpu',all_models_locked=True);persist()
while not idle():time.sleep(30)
state['status']='evaluating';persist();q=queue.Queue()
for i,j in enumerate(jobs):q.put((i,j))
def worker(gpu):
 while True:
  try:i,j=q.get_nowait()
  except queue.Empty:return
  with mutex:state['jobs'][str(i)]={'cell':j['cell'],'profile':j['profile'],'status':'running','gpu':gpu};persist()
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib')
  out=Path(j['base']);out.mkdir(parents=True,exist_ok=True)
  with (out/'evaluation.log').open('a') as log:
   rc=subprocess.run(['taskset','-c',str(80+gpu),sys.executable,'-u',str(p/'evaluate.py'),'--work',str(p),'--job',str(i),'--repeats','3'],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
  with mutex:
   state['jobs'][str(i)].update(status='complete' if rc==0 else 'failed',returncode=rc)
   if rc:state['errors'].append({'job':i,'returncode':rc})
   persist()
  subprocess.run([sys.executable,str(p/'build_report.py')],check=True,stdout=subprocess.DEVNULL)
  if rc:return
with cf.ThreadPoolExecutor(max_workers=2) as pool:
 for f in [pool.submit(worker,g) for g in [2,5]]:
  try:f.result()
  except Exception as e:
   with mutex:state['errors'].append({'error':repr(e)});persist()
state['status']='verifying';persist()
rc=subprocess.run([sys.executable,str(p/'verify.py')]).returncode
state['status']='complete' if rc==0 and not state['errors'] else 'incomplete';state['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z');persist()
subprocess.run([sys.executable,str(p/'build_report.py')],check=True)
