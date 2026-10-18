"""Resume all frozen evaluations after the numerical parity fallback repair."""
from pathlib import Path
import concurrent.futures as cf
import fcntl
import os
import subprocess
import sys
import threading
import time
from common8 import read,write

work=Path(__file__).resolve().parent
guard=open(work/'evaluation_recovery.lock','w')
fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
jobs=read(work/'locked_protocol.json')['jobs'];lock=threading.Lock();claimed=set();errors=[]
state=dict(status='running',pid=os.getpid(),reason='Resume identical frozen evaluations with explicit numerical-equivalence fallback; never change input/model/reset')
write(work/'evaluation_recovery_state.json',state)


def worker(gpu):
    while True:
        with lock:
            completed={i for i,j in enumerate(jobs) if (Path(j['base'])/'evaluation/complete.json').exists()}
            pending=[i for i,j in enumerate(jobs) if i not in completed|claimed and (Path(j['base'])/'test/complete.json').exists()]
            if len(completed)==12:return
            i=pending[0] if pending else None
            if i is not None:claimed.add(i)
        if i is None:
            if errors:return
            time.sleep(5);continue
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib')
        with (work/'logs'/f'evaluate_recovery_{i:02}.log').open('a') as log:
            rc=subprocess.run(['taskset','-c',str(80+gpu),sys.executable,'-u',str(work/'evaluate.py'),'--work',str(work),'--job',str(i),'--repeats','3'],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        with lock:
            if rc:errors.append(dict(job=i,returncode=rc))
            state.update(completed=len(list(work.glob('cells/*/*/evaluation/complete.json'))),errors=errors)
            write(work/'evaluation_recovery_state.json',state)
        if rc:return


with cf.ThreadPoolExecutor(max_workers=5) as pool:
    list(pool.map(worker,[1,3,4,6,7]))
if errors:
    state['status']='failed';write(work/'evaluation_recovery_state.json',state)
    raise RuntimeError(errors)
subprocess.run([sys.executable,str(work/'build_report.py'),'--work',str(work)],check=True)
state['status']='complete';write(work/'evaluation_recovery_state.json',state)
# The original controller finishes its already-running reference jobs. Its
# historical failure remains archived; verification.json is the final gate.
