"""Finite independent CoreNEURON CPU1 measurements on disjoint physical cores."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from screen import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--profile',required=True);p.add_argument('--cpu',type=int,required=True)
    a=p.parse_args();assert 56<=a.cpu<=63
    work=Path(__file__).resolve().parent
    root=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
    data=Path('/data/lcm/DBNN_multineuron_20261002')
    out=work/'results'/a.model/a.profile/'parallel_core';out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(model=a.model,profile=a.profile,cpu=a.cpu,pid=os.getpid(),
               started=time.strftime('%Y-%m-%dT%H:%M:%S%z'),status='waiting_reference')
    manifest=data/a.model/a.profile/'test/manifest.json'
    while not manifest.exists():
        write(out/'state.json',state)
        failed=work/'results'/a.model/'state.json'
        if failed.exists() and json.loads(failed.read_text())['status']=='failed':raise RuntimeError(failed.read_text())
        time.sleep(15)
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for dt in (.05,.1,.25,.5,1.):
        state.update(status='running',dt_ms=dt,updated=time.strftime('%Y-%m-%dT%H:%M:%S%z'));write(out/'state.json',state)
        cmd=[sys.executable,'-u',str(work/'solver_benchmark.py'),'--root',str(root),'--work',str(work),
             '--model',a.model,'--profile',a.profile,'--manifest',str(manifest),'--engine','core',
             '--dt',str(dt),'--repeats','5','--core-cpu',str(a.cpu)]
        with open(out/f'dt{dt}.log','a') as f:rc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        if rc not in (0,2):
            state.update(status='failed',return_code=rc);write(out/'state.json',state);raise SystemExit(rc)
    state.update(status='complete',finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'));write(out/'state.json',state)

if __name__=='__main__':main()
