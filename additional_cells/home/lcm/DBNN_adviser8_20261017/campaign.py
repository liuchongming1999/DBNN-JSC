"""Finite campaign: frozen diagnostics, full-budget fits, locked references/evaluation."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import concurrent.futures as cf
import csv
import fcntl
import os
import queue
import subprocess
import sys
import threading
import time
from common8 import read,write,sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True)
    ap.add_argument('--gpus',default='1,3,4,6,7');a=ap.parse_args()
    work=a.work.resolve();guard=open(work/'campaign.lock','w')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    locked=read(work/'locked_protocol.json');jobs=locked['jobs']
    gpus=list(map(int,a.gpus.split(',')))
    assert 0 not in gpus
    (work/'logs').mkdir(exist_ok=True)
    mutex=threading.Lock();state=dict(status='running',pid=os.getpid(),
        started=datetime.now(timezone.utc).isoformat(),gpus=gpus,jobs={},
        locked_protocol_sha256=sha(work/'locked_protocol.json'),primary_model_unchanged=True)

    def status(key,value):
        with mutex:
            state['jobs'][key]=value;state['updated']=datetime.now(timezone.utc).isoformat()
            write(work/'campaign_state.json',state)

    def run(key,script,args,gpu=None):
        status(key,dict(status='running',gpu=gpu))
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib')
        if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
        cmd=[sys.executable,'-u',str(work/script),*map(str,args)]
        if gpu is not None:cmd=['taskset','-c',str(80+gpu),*cmd]
        with open(work/'logs'/f'{key}.log','a') as log:
            rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        status(key,dict(status='complete' if rc==0 else 'failed',returncode=rc,gpu=gpu))
        if rc:raise RuntimeError((key,rc))

    # Repeated resource checks prevent launching onto another active CUDA job.
    for _ in range(3):
        rows={int(r[0]):(int(r[1]),int(r[2])) for r in csv.reader(subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True).splitlines())}
        assert all(rows[g][0]<=5 and rows[g][1]<128 for g in gpus),rows
        time.sleep(2)

    refq=queue.Queue();trainq=queue.Queue()
    for i in range(len(jobs)):refq.put(i);trainq.put(i)
    error=[]

    def references():
        while True:
            try:i=refq.get_nowait()
            except queue.Empty:return
            run(f'reference_{i:02}','sample_references.py',['--work',work,'--job',i,'--workers',8])

    def train(gpu):
        # One smoke run per worker in its own directory: it cannot become a candidate.
        if gpu==gpus[0]:
            run('frozen_primary','frozen_diagnostics.py',['--root','/home/lcm/DBNN_L5PC_surrogate_20260928',
                '--work',work,'--bootstraps',10000],gpu)
        first=True
        while True:
            try:i=trainq.get_nowait()
            except queue.Empty:break
            j=jobs[i];base=Path(j['base']);condition=read(base/'reference_condition.json')
            assert sha(j['training_manifest'])==j['training_manifest_sha256']
            args=['--root','/home/lcm/DBNN_L5PC_surrogate_20260928','--work',base,
                '--window',500,'--burn',150,'--allow-diagonal','--manifest',j['training_manifest'],
                '--rest',condition['operating_point']['rest_mv']]
            if first:
                run(f'smoke_{i:02}','train_full_matrix.py',args+['--name','preflight','--smoke',
                    '--epochs',1,'--refine-epochs',1],gpu);first=False
            run(f'train_{i:02}','train_full_matrix.py',args+['--name','training'],gpu)
        # All remaining evaluations are claimed under a shared queue after training.
        while True:
            with mutex:
                pending=[i for i,j in enumerate(jobs) if
                    (Path(j['base'])/'training/complete.json').exists() and
                    (Path(j['base'])/'test/complete.json').exists() and i not in claimed]
                if pending:
                    i=pending[0];claimed.add(i)
                else:i=None
                done=len(claimed)==len(jobs)
                all_trained=all((Path(j['base'])/'training/complete.json').exists() for j in jobs)
            if i is not None:
                run(f'evaluate_{i:02}','evaluate.py',['--work',work,'--job',i,'--repeats',3],gpu)
            elif done:return
            elif error:return
            else:time.sleep(5)

    claimed=set()
    try:
        with cf.ThreadPoolExecutor(max_workers=len(gpus)+2) as pool:
            futures=[pool.submit(references) for _ in range(2)]+[pool.submit(train,g) for g in gpus]
            for future in cf.as_completed(futures):
                try:future.result()
                except Exception as exc:
                    error.append(repr(exc));status('error',error)
        if error:raise RuntimeError(error)
        assert all((Path(j['base'])/'evaluation/complete.json').exists() for j in jobs)
        run('report','build_report.py',['--work',work])
        state['status']='complete'
    except Exception as exc:
        state['status']='failed';state['error']=repr(exc);raise
    finally:
        state['updated']=datetime.now(timezone.utc).isoformat();write(work/'campaign_state.json',state)


if __name__=='__main__':main()
