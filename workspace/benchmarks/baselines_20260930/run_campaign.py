"""Persistent experiment queues; no scheduler jobs or unrelated processes touched."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from panel import save

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--worker',choices=['tcn','recurrent','reduce','pca']);p.add_argument('--launch',action='store_true');a=p.parse_args()
    root=a.root.resolve();work=a.work.resolve();data=a.data.resolve();os.chdir(work);logs=work/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(PYTHONPATH=str(work/'deps'),LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',PYTHONFAULTHANDLER='1')
    if a.worker=='pca':env.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    if a.launch:
        workers={'pca':'90-93','reduce':'81','tcn':'80','recurrent':'82'};started=[]
        for w,cpus in workers.items():
            pidfile=logs/f'{w}_pid.json'
            if pidfile.exists():
                pid=json.loads(pidfile.read_text())['pid'];proc=Path(f'/proc/{pid}/cmdline')
                if proc.exists() and b'run_campaign.py' in proc.read_bytes():started.append(dict(worker=w,pid=pid,already_running=True));continue
            e=env.copy()
            if w=='pca':e.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
            log=(logs/f'{w}_controller.log').open('a')
            cmd=['taskset','-c',cpus,sys.executable,'-u',str(Path(__file__).resolve()),'--root',str(root),'--work',str(work),'--data',str(data),'--worker',w]
            c=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=e,start_new_session=True);rec=dict(worker=w,pid=c.pid,cpus=cpus,gpu='0 shared' if w in ['tcn','recurrent'] else None,started=time.strftime('%Y-%m-%dT%H:%M:%S%z'));save(pidfile,rec);started.append(rec)
        save(work/'campaign.json',dict(started=started,training='Default author recipes, no training/capacity sweeps; TCN and recurrent queues share GPU0, CPU1 reduction uses affinity81.',resource_status='Shared GPU/CPU; no existing process stopped. Publication timing is a separate phase.'));print(json.dumps(started,indent=2));return
    def run(label,cmd):
        save(logs/f'{a.worker}_state.json',dict(status='running',job=label,updated=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        print('START',label,flush=True)
        with (logs/(label+'.log')).open('a') as log:r=subprocess.run([sys.executable,'-u']+cmd,stdout=log,stderr=subprocess.STDOUT,env=env).returncode
        if r:
            save(logs/f'{a.worker}_state.json',dict(status='failed',job=label,returncode=r,log=str(logs/(label+'.log'))));raise SystemExit(r)
        print('FINISH',label,flush=True)
    if a.worker=='pca':
        for dt in [.5,.25]:run(f'pca_{dt:g}ms',['prepare_pca.py','--root',str(root),'--aux',str(data/'tcn_aux'),'--dt',str(dt)])
    elif a.worker=='reduce':
        for dt in [1,.5,.25]:run(f'reduce_{dt:g}ms',['benchmark_reduce.py','--root',str(root),'--work',str(work),'--dt',str(dt),'--repeats','30'])
    else:
        methods=['tcn'] if a.worker=='tcn' else ['elm','lstm']
        for dt in [1,.5,.25]:
            for method in methods:
                if method=='tcn':
                    while not (data/'tcn_aux'/f'pca_{dt:g}ms'/'complete.json').exists():time.sleep(10)
                args=['train_default.py','--root',str(root),'--work',str(work),'--data',str(data),'--method',method,'--dt',str(dt)]
                run(f'{method}_{dt:g}ms_smoke',args+['--smoke'])
                run(f'{method}_{dt:g}ms',args)
                run(f'{method}_{dt:g}ms_evaluate',['benchmark_models.py','--root',str(root),'--work',str(work),'--method',method,'--dt',str(dt)])
    save(logs/f'{a.worker}_state.json',dict(status='complete',updated=time.strftime('%Y-%m-%dT%H:%M:%S%z')))

if __name__=='__main__':main()
