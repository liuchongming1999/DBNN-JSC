"""Finite, resource-checked event-training campaign on currently idle GPUs."""
import argparse
import csv
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from train_events import read, write

WORK = Path(__file__).resolve().parent
ROOT = Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
PRIORITY = [('golgi','ampa'), ('golgi','ampa_nmda'), ('basket','ampa'), ('basket','ampa_nmda'),
            ('ca1','ampa_nmda'), ('hay_l5','ampa'), ('human_l23','ampa'), ('granule','ampa_nmda'),
            ('ca1','ampa'), ('hay_l5','ampa_nmda'), ('granule','ampa'), ('human_l23','ampa_nmda')]


def origin(cell):
    tag = 'DBNN_multineuron_20261002' if cell in ('granule','basket','golgi') else 'DBNN_noncerebellar_20261002'
    return Path('/home/lcm')/tag, Path('/data/lcm')/tag


def resources():
    def query(flag):
        return subprocess.run(['nvidia-smi', flag, '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=15).stdout
    values = {}
    for row in csv.reader(query('--query-gpu=index,uuid,utilization.gpu,memory.used').splitlines()):
        index, uuid, util, mem = [x.strip() for x in row]
        values[int(index)] = dict(index=int(index), uuid=uuid, utilization=int(util), memory_mib=int(mem), processes=[])
    by_uuid = {r['uuid']: r for r in values.values()}
    for row in csv.reader(query('--query-compute-apps=gpu_uuid,pid').splitlines()):
        if len(row) != 2:
            continue
        uuid, pid = [x.strip() for x in row]
        if uuid not in by_uuid:
            continue
        try:
            proc = Path('/proc')/pid
            owner = proc.stat().st_uid
            command = proc.joinpath('cmdline').read_bytes().decode().replace('\0',' ')
        except OSError:
            owner, command = None, 'Process disappeared during snapshot'
        by_uuid[uuid]['processes'].append(dict(pid=int(pid), uid=owner, command=command))
    return values


def idle(record):
    return not record['processes'] and record['utilization'] <= 5 and record['memory_mib'] < 128


def folder(cell, profile):
    return WORK/'results'/cell/profile


def complete(cell, profile):
    training = folder(cell, profile)/'dt1.0ms/final_selection.json'
    gpu = folder(cell, profile)/'gpu/complete.json'
    return training.exists() and gpu.exists() and read(training)['finished'] and read(gpu)['finished']


def worker(args):
    out = folder(args.cell, args.profile)
    out.mkdir(parents=True, exist_ok=True)
    guards = []
    try:
        for path in [out/'campaign.lock', WORK/f'gpu{args.gpu}.lock', Path('/home/lcm/DBNN_noncerebellar_20261002')/f'gpu{args.gpu}.lock']:
            handle = path.open('w')
            guards.append(handle)
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observations = []
        for i in range(3):
            resource = resources()[args.gpu]
            observations.append(resource)
            if not idle(resource):
                write(out/'campaign_state.json', dict(status='waiting_resource', gpu=args.gpu, observations=observations))
                return 75
            if i < 2:
                time.sleep(3)
        if complete(args.cell, args.profile):
            return 0
        project, data = origin(args.cell)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=resource['uuid'], OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib')
        base = dict(cell=args.cell, profile=args.profile, gpu=args.gpu, pid=os.getpid(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                    started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), initial_resources=observations)
        for phase, script, extra in [
                ('training', 'train_events.py', ['--project',str(project),'--cell',args.cell,'--profile',args.profile]),
                ('evaluation', 'gpu_evaluate.py', ['--data',str(data),'--model',args.cell,'--profile',args.profile,'--repeats','30'])]:
            write(out/'campaign_state.json', dict(**base, status='running', phase=phase, updated=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            command = [sys.executable, '-u', str(WORK/script), '--root',str(ROOT),'--work',str(WORK),*extra]
            with (out/(phase+'.log')).open('a') as log:
                rc = subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc:
                write(out/'campaign_state.json', dict(**base,status='failed',phase=phase,returncode=rc,updated=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
                return rc
        assert complete(args.cell, args.profile)
        write(out/'campaign_state.json', dict(**base,status='complete',completed=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        return 0
    except BlockingIOError:
        return 75
    finally:
        for handle in guards:
            handle.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--cell')
    parser.add_argument('--profile')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--interval', type=float, default=10.)
    args = parser.parse_args()
    logs = WORK/'logs'
    logs.mkdir(exist_ok=True)
    if args.worker:
        raise SystemExit(worker(args))
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--interval', str(args.interval)]
    if args.launch:
        saved = logs/'controller_pid.json'
        if saved.exists():
            old = read(saved)
            process = Path('/proc')/str(old['pid'])/'cmdline'
            if process.exists() and str(WORK).encode() in process.read_bytes() and b'campaign.py' in process.read_bytes():
                print(dict(already_running=True, **old), flush=True)
                return
        with (logs/'controller.log').open('a') as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(pid=child.pid, started=time.strftime('%Y-%m-%dT%H:%M:%S%z'), jobs=len(PRIORITY), trials=[0.,.2,1.], epochs=[1000,400])
        write(saved, record)
        print(record, flush=True)
        return
    guard = open(logs/'controller.lock', 'w')
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    running, failures, streak = {}, {}, {}
    while True:
        for gpu, job in list(running.items()):
            rc = job['process'].poll()
            if rc is None:
                continue
            job['log'].close()
            if rc not in (0,75):
                failures[job['key']] = rc
            del running[gpu]
        pending = [(c,p) for c,p in PRIORITY if not complete(c,p) and f'{c}/{p}' not in failures]
        try:
            snapshot = resources()
            for gpu, resource in sorted(snapshot.items()):
                if gpu in running or not idle(resource):
                    streak[gpu] = 0
                    continue
                streak[gpu] = streak.get(gpu,0)+1
                if streak[gpu] < 3:
                    continue
                for cell, profile in pending:
                    key = f'{cell}/{profile}'
                    if any(job['key']==key for job in running.values()):
                        continue
                    path = folder(cell, profile)
                    path.mkdir(parents=True, exist_ok=True)
                    with (path/'campaign.lock').open('w') as probe:
                        try:
                            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue
                    log = (logs/(cell+'_'+profile+'_worker.log')).open('a')
                    worker_command = command+['--worker','--cell',cell,'--profile',profile,'--gpu',str(gpu)]
                    child = subprocess.Popen(['taskset','-c',str(80+gpu),*worker_command], stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    running[gpu] = dict(process=child,log=log,key=key)
                    streak[gpu] = 0
                    print('STARTED', key, 'gpu', gpu, 'pid', child.pid, flush=True)
                    break
            jobs = []
            for cell, profile in PRIORITY:
                path = folder(cell, profile)
                state = read(path/'campaign_state.json') if (path/'campaign_state.json').exists() else dict(status='queued')
                progress = path/'dt1.0ms/progress.json'
                selection = path/'dt1.0ms/final_selection.json'
                gpu_result = path/'gpu/complete.json'
                row = {**state, 'cell':cell, 'profile':profile}
                if progress.exists():
                    row['progress'] = read(progress)
                if selection.exists():
                    trained = read(selection)
                    row['selection'] = dict(selected=trained['selected'],original_validation=trained['original_validation'],
                                            locked_at=trained['locked_at'],checkpoint_sha256=trained['checkpoint_sha256'],elapsed_s=trained['elapsed_s'])
                if gpu_result.exists():
                    row['evaluation'] = read(gpu_result)['summary']
                jobs.append(row)
            status = 'running' if running or pending else ('complete_with_failures' if failures else 'complete')
            write(logs/'campaign_status.json', dict(status=status,updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  jobs=jobs,failures=failures,running={str(g):dict(key=j['key'],pid=j['process'].pid) for g,j in running.items()},
                  resources=snapshot,scope='12 per-cell/profile conditions at 1ms; three matched continuing-training trials, validation-only reset/checkpoint selection, original 31-window test after lock'))
            if not running and not pending:
                return
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            write(logs/'controller_error.json', dict(error=repr(error),updated=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
