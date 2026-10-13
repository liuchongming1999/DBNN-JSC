"""Wait for locked checkpoints and re-time both training-window families."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path('/home/lcm/DBNN_L5PC_surrogate_20260928');WORK=ROOT/'benchmarks/fair_window_20261005'
def save(obj):
    p=WORK/'timing_state.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def main():
    guard=open(WORK/'timing.lock','w');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    required=[WORK/'dbnn'/f'dbnn_w{w}_dt{dt:g}ms_burn150/complete.json' for w in [400,500] for dt in [1.,.5,.25]]
    while not all(p.exists() and json.loads(p.read_text())['finished'] for p in required):
        save(dict(status='waiting_for_checkpoints',completed=sum(p.exists() for p in required),total=6));time.sleep(10)
    # GPU3 is used only after its DBNN training worker has completed.
    from campaign import stable_idle
    stable_idle(3)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='3',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib')
    (WORK/'boundary_controls/logs').mkdir(exist_ok=True)
    cpu_guard=open(WORK/'boundary_controls/logs/cpu_evaluation.lock','w');fcntl.flock(cpu_guard,fcntl.LOCK_EX)
    for window in [500,400]:
        subprocess.run([sys.executable,str(WORK/'prepare_benchmark.py'),'--window',str(window)],check=True)
        out=WORK/f'timing_w{window}'
        for label,args in [('gpu',['gpu_benchmark.py','--repeats','30']),('cpu1',['cpu_benchmark.py','--threads','1','--repeats','30']),('cpu8',['cpu_benchmark.py','--threads','8','--repeats','30'])]:
            done=out/label/'complete.json'
            if done.exists() and json.loads(done.read_text())['finished']:continue
            save(dict(status='timing',window_ms=window,backend=label,started=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
            with (out/(label+'.log')).open('a') as log:
                rc=subprocess.run([sys.executable,'-u',str(out/args[0]),'--root',str(ROOT),'--work',str(out)]+args[1:],stdout=log,stderr=subprocess.STDOUT,env=env).returncode
            if rc:save(dict(status='failed',window_ms=window,backend=label,returncode=rc));raise SystemExit(rc)
    save(dict(status='complete',finished=True,completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'),scope='Two short-window families; same checkpoint across3backends;31windows;30repeats; independent FFT/reset and CPU/GPU oracle checks.'))
if __name__=='__main__':main()
