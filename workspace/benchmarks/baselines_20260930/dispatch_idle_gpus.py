"""Run remaining default experiments on idle GPUs; existing queues keep running.

Only LSTM may share explicitly allowed GPUs, and only with one low-load CIFAR
process owned by this user. No foreign process is stopped or moved. This finite
dispatcher exits after all nine experiments have completed their first evaluation.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from job_lock import acquire_lock
from panel import save


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def args_for_job(root, work, data, method, dt):
    return ['--root',str(root),'--work',str(work),'--data',str(data),
            '--method',method,'--dt',str(dt)]


def active_jobs(work):
    active = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            args = proc.joinpath('cmdline').read_bytes().decode().split('\0')
            if not any(Path(a).name in ['train_default.py','benchmark_models.py'] for a in args):
                continue
            if Path(args[args.index('--work')+1]).resolve() != work:
                continue
            method=args[args.index('--method')+1]
            dt=float(args[args.index('--dt')+1])
            active.add(f'{method}_{dt:g}ms')
        except (OSError,ValueError,IndexError):
            continue
    return active


def gpu_snapshot():
    def query(flag):
        return subprocess.run(['nvidia-smi',flag,'--format=csv,noheader,nounits'],
                              check=True,capture_output=True,text=True,timeout=15).stdout
    gpus=[]
    for row in csv.reader(query('--query-gpu=index,uuid,utilization.gpu,memory.used').splitlines()):
        index,uuid,util,memory=[v.strip() for v in row]
        gpus.append(dict(index=int(index),uuid=uuid,utilization=int(util),memory_mib=int(memory),processes=[]))
    lookup={g['uuid']:g for g in gpus}
    for row in csv.reader(query('--query-compute-apps=gpu_uuid,pid').splitlines()):
        if len(row)!=2:
            continue
        uuid,pid=[v.strip() for v in row]
        if uuid not in lookup:
            continue
        proc=Path('/proc')/pid
        try:
            uid=proc.stat().st_uid
            cmd=proc.joinpath('cmdline').read_bytes().decode().replace('\0',' ')
        except OSError:
            uid,cmd=None,''
        lookup[uuid]['processes'].append(dict(pid=int(pid),uid=uid,command=cmd))
    return gpus


def eligibility(gpu, shared_gpus, uid):
    if not gpu['processes'] and gpu['utilization']<=5 and gpu['memory_mib']<128:
        return 'idle'
    if gpu['index'] in shared_gpus and gpu['utilization']<=45 and gpu['memory_mib']<=2048 and len(gpu['processes'])==1:
        p=gpu['processes'][0]
        if p['uid']==uid and 'cifar100_depth_margin_20260927/train.py' in p['command']:
            return 'shared_low_load'
    return None


def finished(folder):
    lock=read(folder/'complete.json');result=read(folder/'evaluation/complete.json')
    return bool(lock and lock.get('default_budget_completed') and result and result.get('finished')
                and result['checkpoint']['checkpoint_sha256']==lock['checkpoint_sha256'])


def worker(a):
    job=f'{a.method}_{a.dt:g}ms';folder=a.work/'aligned'/job
    guard=acquire_lock(folder/'training.lock',inherited_env='DBNN_TRAIN_LOCK_FD')
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu),DBNN_TRAIN_LOCK_FD=str(guard.fileno()),
               LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib',
               PYTHONPATH=str(a.work/'deps'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    os.chdir(a.work)
    state_path=a.work/'logs'/f'dispatched_{job}_state.json'
    base=dict(job=job,gpu=a.gpu,resource_mode=a.resource_mode,pid=os.getpid(),
              cpu_affinity=sorted(os.sched_getaffinity(0)),started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    def run(phase,cmd):
        save(state_path,dict(**base,status='running',phase=phase))
        with (a.work/'logs'/f'dispatched_{job}_{phase}.log').open('a') as log:
            rc=subprocess.run(cmd,env=env,pass_fds=(guard.fileno(),),stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:
            save(state_path,dict(**base,status='failed',phase=phase,returncode=rc))
            raise SystemExit(rc)
    if not (folder/'complete.json').exists():
        cmd=[sys.executable,'-u',str(a.work/'train_default.py')]+args_for_job(a.root,a.work,a.data,a.method,a.dt)
        run('smoke',cmd+['--smoke'])
        run('training',cmd)
    run('evaluation',[sys.executable,'-u',str(a.work/'benchmark_models.py'),'--root',str(a.root),
                     '--work',str(a.work),'--method',a.method,'--dt',str(a.dt)])
    save(state_path,dict(**base,status='complete',completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--launch',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--method');p.add_argument('--dt',type=float);p.add_argument('--gpu',type=int);p.add_argument('--resource-mode')
    p.add_argument('--interval',type=float,default=20);p.add_argument('--stable-checks',type=int,default=3);p.add_argument('--allow-shared-gpus',default='')
    a=p.parse_args();a.root=a.root.resolve();a.work=a.work.resolve();a.data=a.data.resolve();logs=a.work/'logs';logs.mkdir(exist_ok=True)
    if a.worker:
        worker(a);return
    command=[sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(a.root),'--work',str(a.work),'--data',str(a.data),
             '--interval',str(a.interval),'--stable-checks',str(a.stable_checks),'--allow-shared-gpus',a.allow_shared_gpus]
    if a.launch:
        old=read(logs/'dispatcher_pid.json')
        if old:
            proc=Path('/proc')/str(old['pid'])/'cmdline'
            if proc.exists() and b'dispatch_idle_gpus.py' in proc.read_bytes():
                print(json.dumps(dict(already_running=True,**old)));return
        with (logs/'dispatcher_controller.log').open('a') as log:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(pid=child.pid,started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),allow_shared_gpus=a.allow_shared_gpus,stable_checks=a.stable_checks,interval_s=a.interval)
        save(logs/'dispatcher_pid.json',record);print(json.dumps(record));return
    controller_guard=acquire_lock(logs/'dispatcher.lock',blocking=False)
    shared={int(v) for v in a.allow_shared_gpus.split(',') if v};streaks={};running={};failed=set()
    # Light LSTM first; remaining idle GPUs take the longest waiting TCN/ELM jobs.
    priority=[('lstm',1.),('tcn',.25),('elm',.5),('lstm',.5),('elm',.25),('lstm',.25),('tcn',.5),('elm',1.),('tcn',1.)]
    while True:
        for gpu,rec in list(running.items()):
            rc=rec['process'].poll()
            if rc is not None:
                if rc:failed.add(rec['job'])
                rec['log'].close();del running[gpu]
        pending=[(m,dt) for m,dt in priority if not finished(a.work/'aligned'/f'{m}_{dt:g}ms')]
        if not pending and not running:
            save(logs/'dispatcher_state.json',dict(status='complete',updated=time.strftime('%Y-%m-%dT%H:%M:%S%z')));return
        try:
            gpus=gpu_snapshot();active=active_jobs(a.work)
            memory_kib=int(next(v.split()[1] for v in Path('/proc/meminfo').read_text().splitlines() if v.startswith('MemAvailable:')))
            for gpu in sorted(gpus,key=lambda g:(bool(g['processes']),g['utilization'],g['index'])):
                index=gpu['index'];mode=eligibility(gpu,shared,os.getuid())
                if index in running or mode is None:
                    streaks[index]=(None,0);continue
                previous,count=streaks.get(index,(None,0));count=count+1 if previous==mode else 1;streaks[index]=(mode,count)
                if count<a.stable_checks or memory_kib<25*1024**2:
                    continue
                for method,dt in pending:
                    job=f'{method}_{dt:g}ms'
                    if job in active or job in failed or any(v['job']==job for v in running.values()):continue
                    if mode=='shared_low_load' and method!='lstm':continue
                    folder=a.work/'aligned'/job
                    try:guard=acquire_lock(folder/'training.lock',blocking=False)
                    except BlockingIOError:continue
                    try:
                        # A legacy queue may have completed this job during the query.
                        if finished(folder):continue
                        env=os.environ.copy();env['DBNN_TRAIN_LOCK_FD']=str(guard.fileno())
                        cmd=command+['--worker','--method',method,'--dt',str(dt),'--gpu',str(index),'--resource-mode',mode]
                        log=(logs/f'dispatched_{job}_worker.log').open('a')
                        child=subprocess.Popen(['taskset','-c',str(84+index)]+cmd,env=env,stdin=subprocess.DEVNULL,
                                               stdout=log,stderr=subprocess.STDOUT,pass_fds=(guard.fileno(),),start_new_session=True)
                        running[index]=dict(process=child,log=log,job=job,resource_mode=mode)
                        print(json.dumps(dict(started=job,gpu=index,pid=child.pid,resource_mode=mode,observed_gpu=gpu)),flush=True)
                        memory_kib-=25*1024**2;streaks[index]=(None,0);break
                    finally:guard.close()
            state=dict(status='running',updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                       policy='Idle GPU first; 3 stable checks. Explicit GPU1 may share one owned CIFAR process at <=45% load with LSTM only. Existing experiments continue unchanged.',
                       active_jobs=sorted(active),pending=[f'{m}_{dt:g}ms' for m,dt in pending],failed=sorted(failed),
                       running={str(g):dict(job=r['job'],pid=r['process'].pid,resource_mode=r['resource_mode']) for g,r in running.items()},gpus=gpus)
            save(logs/'dispatcher_state.json',state)
        except (OSError,subprocess.SubprocessError,ValueError) as e:
            save(logs/'dispatcher_state.json',dict(status='resource_query_retry',error=repr(e),updated=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        time.sleep(a.interval)


if __name__=='__main__':main()
