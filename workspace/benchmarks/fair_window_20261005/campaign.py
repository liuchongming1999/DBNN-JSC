"""One persistent queue per idle GPU for the authorized finite experiment grid."""
import argparse, fcntl, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path('/home/lcm/DBNN_L5PC_surrogate_20260928')
WORK=ROOT/'benchmarks/fair_window_20261005'
DATA=Path('/data/lcm/DBNN_baselines_20260930')
JOBS={0:[('dbnn',400,.5),('tcn',None,.25)],
      1:[('dbnn',500,.5),('elm',None,.25)],
      2:[('dbnn',400,.25)],3:[('dbnn',500,.25)],
      4:[('tcn',None,1.)],5:[('tcn',None,.5)],
      6:[('elm',None,1.)],7:[('elm',None,.5)]}
def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2)+'\n');q.replace(p)
def stable_idle(gpu):
    ok=0
    while ok<3:
        out=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
        util,mem=map(int,out.strip().split(','));ok=ok+1 if util<=5 and mem<=256 else 0
        if ok<3:time.sleep(3 if ok else 20)
def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',type=int);p.add_argument('--launch',action='store_true');a=p.parse_args()
    logs=WORK/'logs';logs.mkdir(exist_ok=True)
    if a.launch:
        started=[]
        for gpu in JOBS:
            pidfile=logs/f'gpu{gpu}_pid.json'
            if pidfile.exists():
                old=json.loads(pidfile.read_text());proc=Path(f"/proc/{old['pid']}/cmdline")
                if proc.exists() and b'fair_window_20261005/campaign.py' in proc.read_bytes():
                    started.append(old);continue
            env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib',PYTHONPATH=str(WORK/'boundary_controls/deps'))
            with (logs/f'gpu{gpu}_controller.log').open('a') as log:
                child=subprocess.Popen(['taskset','-c',str(gpu),sys.executable,'-u',str(Path(__file__).resolve()),'--worker',str(gpu)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
            rec=dict(pid=child.pid,gpu=gpu,cpu_affinity=[gpu],started=time.strftime('%Y-%m-%dT%H:%M:%S%z'));save(pidfile,rec);started.append(rec)
        save(WORK/'campaign.json',dict(started=started,jobs=JOBS,recipe='Random-init DBNN400/500ms,150ms loss burn; TCN/ELM same150ms loss burn controls with author optimizer/budgets. Author-default baseline checkpoints retained. LSTM already500/150ms reused.',one_ms_reuse='Completed matched-random-init20261004 arms; no retraining or selected-checkpoint changes. Fine sampling uses exactly each baseline source panel.'))
        print(json.dumps(started,indent=2));return
    gpu=a.worker;guard=open(logs/f'gpu{gpu}.lock','w');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    statefile=logs/f'gpu{gpu}_state.json'
    def run(label,args):
        save(statefile,dict(status='running',job=label,command=args,updated=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        print('START',label,flush=True)
        with (logs/(label+'.log')).open('a') as log:
            rc=subprocess.run([sys.executable,'-u']+args,stdout=log,stderr=subprocess.STDOUT,cwd=WORK).returncode
        if rc:
            save(statefile,dict(status='failed',job=label,returncode=rc));raise SystemExit(rc)
        print('FINISH',label,flush=True)
    for method,window,dt in JOBS[gpu]:
        save(statefile,dict(status='waiting_for_idle_gpu',gpu=gpu,job=[method,window,dt]));stable_idle(gpu)
        if method=='dbnn':
            args=[str(WORK/'train_dbnn_windows.py'),'--root',str(ROOT),'--work',str(WORK/'dbnn'),'--window',str(window),'--burn','150','--dt',str(dt)]
            run(f'dbnn{window}_{dt:g}ms_smoke',args+['--smoke','--epochs','2','--refine-epochs','2'])
            run(f'dbnn{window}_{dt:g}ms',args)
        else:
            bw=WORK/'boundary_controls'
            if method=='elm':
                run(f'elm_{dt:g}ms_parity',[str(bw/'probe_boundary.py'),'--root',str(ROOT),'--work',str(bw),'--dt',str(dt)])
            args=[str(bw/'train_default.py'),'--root',str(ROOT),'--work',str(bw),'--data',str(DATA),'--method',method,'--dt',str(dt),'--loss-burn-ms','150']
            if method=='elm':args+=['--elm-cuda-graph']
            run(f'{method}_{dt:g}ms_smoke',args+['--smoke'])
            run(f'{method}_{dt:g}ms',args)
            run(f'{method}_{dt:g}ms_evaluate',[str(bw/'benchmark_models.py'),'--root',str(ROOT),'--work',str(bw),'--method',method,'--dt',str(dt),'--repeats','30'])
    save(statefile,dict(status='complete',gpu=gpu,finished=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
if __name__=='__main__':main()
