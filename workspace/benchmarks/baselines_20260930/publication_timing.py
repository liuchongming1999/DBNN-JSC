"""Refresh all methods in one timing phase after the training queues finish."""
import argparse,json,os,random,shutil,subprocess,sys,time
from pathlib import Path
from panel import save

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args();root=a.root.resolve();work=a.work.resolve();out=work/'publication';out.mkdir(exist_ok=True)
    # This is the last phase of this experiment, not a scheduled reminder.
    required=[work/'logs'/f'{w}_state.json' for w in ['tcn','recurrent','reduce']]
    while True:
        states=[json.loads(f.read_text()) if f.exists() else {} for f in required]
        if any(s.get('status')=='failed' for s in states):save(out/'state.json',dict(status='blocked_by_failed_experiment',workers=states));raise SystemExit(1)
        if all(s.get('status')=='complete' for s in states):break
        save(out/'state.json',dict(status='waiting_for_default_training_and_reduction',workers=states));time.sleep(30)
    # Reuse the already validated DBNN implementation and exactly its selected
    # checkpoints. Outputs/policies go to a separate timing directory.
    src=root/'benchmarks/resolution_frontier_20260929';dbnn=out/'dbnn';dbnn.mkdir(exist_ok=True)
    for name in ['gpu_benchmark.py','gpu_backend.py','cpu_benchmark.py','common.py','specs.py']:
        shutil.copy2(src/name,dbnn/name)
    target=dbnn/'b6_warm_0p25ms'
    if not target.exists():target.symlink_to(src/'b6_warm_0p25ms',target_is_directory=True)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',LD_LIBRARY_PATH='/home/lcm/miniconda3/envs/torch/lib',PYTHONPATH=str(work/'deps'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    def run(label,cmd,completion=None):
        if completion is not None and completion.exists():
            prior=json.loads(completion.read_text())
            if prior.get('finished') and len(prior.get('records',[]))==31:
                if not all(r.get('validation',{}).get('passed',True) for r in prior['records']):
                    raise RuntimeError(('Existing output failed validation',str(completion)))
                print('REUSE_COMPLETE',label,str(completion),flush=True)
                return
        save(out/'state.json',dict(status='timing',method=label,started=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
        with (out/(label+'.log')).open('a') as log:rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:save(out/'state.json',dict(status='failed',method=label,returncode=rc));raise SystemExit(rc)
    run('dbnn_gpu',[sys.executable,str(dbnn/'gpu_benchmark.py'),'--root',str(root),'--work',str(dbnn),'--repeats','30'],dbnn/'gpu/complete.json')
    for nt in [1,8]:run(f'dbnn_cpu{nt}',[sys.executable,str(dbnn/'cpu_benchmark.py'),'--root',str(root),'--work',str(dbnn),'--threads',str(nt),'--repeats','30'],dbnn/f'cpu{nt}/complete.json')
    jobs=[(m,dt) for m in ['tcn','elm','lstm'] for dt in [1.,.5,.25]];random.Random(930301).shuffle(jobs)
    for method,dt in jobs:run(f'{method}_{dt:g}ms',[sys.executable,str(work/'benchmark_models.py'),'--root',str(root),'--work',str(work),'--method',method,'--dt',str(dt),'--publication','--repeats','30'],work/'aligned'/f'{method}_{dt:g}ms/publication/complete.json')
    # NEURON and CoreNEURON use the exact original selected-event manifest.
    manifest=root/'benchmarks/neuron_single_expanded_20260929/selected_20260929/manifest.json';solver=root/'benchmarks/neuron_single_expanded_20260929'
    for dt in [1.,.5,.25]:
        (out/'neuron8').mkdir(exist_ok=True);(out/'core1').mkdir(exist_ok=True)
        run(f'neuron8_{dt:g}ms',[sys.executable,str(solver/'run_multisplit_grid.py'),'--root',str(root),'--manifest',str(manifest),'--output-dir',str(out/'neuron8'),'--previous-results',str(solver/'solver_results'), '--threads','8','--cpus','72,73,74,75,76,77,78,79','--dt',str(dt),'--repeats','30'],out/f'neuron8/status_nt8_dt{dt:g}.json')
        # The established Core runner loads the compiled custom mechanisms,
        # removes dormant source-event overhead and checks every case against
        # the identical-dt NEURON voltage and event times.
        run(f'core1_{dt:g}ms',[sys.executable,str(root/'benchmarks/core_threading_20260929/run_core_threads.py'),'--root',str(root),'--manifest',str(manifest),'--output-dir',str(out/'core1'),'--previous-results',str(solver/'solver_results'),'--threads','1','--cpus','72','--dt',str(dt),'--repeats','30'],out/f'core1/status_nt1_dt{dt:g}.json')
        run(f'reduce1_{dt:g}ms',['taskset','-c','72',sys.executable,str(work/'benchmark_reduce.py'),'--root',str(root),'--work',str(work),'--dt',str(dt),'--publication','--repeats','30'],out/f'reduce_{dt:g}ms/complete.json')
    save(out/'state.json',dict(status='complete',finished=True,completed=time.strftime('%Y-%m-%dT%H:%M:%S%z'),scope='All default checkpoints locked; contemporaneous timing phase; 30 repeated full prepared compute calls, shared-server status retained. No claim of exclusive hardware.'))

if __name__=='__main__':main()
