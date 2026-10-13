"""Finite independent reference preparation overlapping GPU fitting, with locks."""
import argparse,fcntl,json,os,subprocess,sys,time
from pathlib import Path
from screen import write

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
    work=Path(__file__).resolve().parent;data=Path('/data/lcm/DBNN_multineuron_20261002');out=work/'results'/a.model;out.mkdir(exist_ok=True)
    lock=open(out/'presampling.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    env=os.environ.copy();env.update(LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    op=out/'screen/operating_point.json'
    while not op.exists():
        state=json.loads((out/'state.json').read_text())
        if state['status']=='failed':raise RuntimeError(state)
        time.sleep(10)
    for profile,kind,dt in [('ampa','train',1.),('ampa','train',.5),('ampa','train',.25),('ampa','test',1.),('ampa_nmda','train',1.),('ampa_nmda','test',1.)]:
        write(out/'presampling_state.json',dict(status='running',profile=profile,kind=kind,dt_ms=dt,workers=a.workers,pid=os.getpid()))
        cmd=[sys.executable,'-u',str(work/'sample_panel.py'),'--work',str(work),'--data',str(data),'--model',a.model,'--profile',profile,'--kind',kind,'--dt',str(dt),'--workers',str(a.workers)]
        with open(out/f'presample_{profile}_{kind}_{dt}.log','a') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    write(out/'presampling_state.json',dict(status='complete',workers=a.workers))

if __name__=='__main__':main()
