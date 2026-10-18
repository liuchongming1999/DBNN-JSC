from pathlib import Path
import subprocess,os,time,json
w=Path(__file__).resolve().parent
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
with (w/'timing_recheck_20261017.log').open('w') as log, (w/'timing_recheck_gpu_20261017.jsonl').open('w') as mon:
 p=subprocess.Popen(['/home/lcm/DBNN_L5PC_surrogate_20260928/.venv/bin/python','-u',str(w/'final_gpu_recheck_20261017.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
 (w/'timing_recheck_child.pid').write_text(str(p.pid))
 while p.poll() is None:
  gpu=subprocess.run(['nvidia-smi','--id=0','--query-gpu=uuid,utilization.gpu,memory.used,power.draw,clocks.sm','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
  apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
  mon.write(json.dumps(dict(t=time.time(),gpu=gpu,apps=apps,child=p.pid))+'\n');mon.flush();time.sleep(2)
 (w/'timing_recheck_exit_20261017.json').write_text(json.dumps(dict(returncode=p.returncode,finished_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'))))
